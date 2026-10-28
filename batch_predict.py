"""
批量推理脚本 —— 3060 单卡
用法:
    python batch_predict.py \
        --model checkpoints/checkpoint_epoch20.pth \
        --test-dir ./data/test_hq \
        --output submission.csv \
        --batch-size 32 \
        --scale 0.5 \
        --bilinear

也可以直接读 test.zip:
    python batch_predict.py \
        --model checkpoints/checkpoint_epoch20.pth \
        --test-zip /path/to/test.zip \
        --output submission.csv
"""
import argparse
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset


def rle_encode(mask, order='F'):
    """Carvana 官方要求列优先 (Fortran order) RLE 编码"""
    pixels = mask.flatten(order=order)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_args():
    parser = argparse.ArgumentParser(description='Batch predict masks → submission.csv')
    parser.add_argument('--model', '-m', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--test-dir', '-d', help='Directory of test images')
    parser.add_argument('--test-zip', '-z', help='Zip file of test images')
    parser.add_argument('--output', '-o', default='submission.csv', help='Output CSV path')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor (must match training)')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling (must match training)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision for inference')
    return parser.parse_args()


def load_test_images_from_dir(test_dir):
    """从目录读图片，返回 (文件名, PIL.Image) 迭代器"""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = sorted(
        f for f in os.listdir(test_dir)
        if Path(f).suffix.lower() in exts and not f.startswith('.')
    )
    for f in files:
        img = Image.open(os.path.join(test_dir, f)).convert('RGB')
        yield f, img


def load_test_images_from_zip(zip_path):
    """从 zip 读图片，返回 (文件名, PIL.Image) 迭代器"""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        img_files = sorted([
            f for f in zf.namelist()
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            and not f.startswith('__MACOSX')
        ])
        for img_path in img_files:
            with zf.open(img_path) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
            yield os.path.basename(img_path), img


def main():
    args = get_args()

    # ---- 设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', end='')
    # if device.type == 'cuda':
    #     # print(f' | {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    # else:
    #     print()

    # ---- 加载权重 ----
    print(f'加载权重: {args.model}')
    state_dict = torch.load(args.model, map_location=device, weights_only=False)
    state_dict.pop('mask_values', None)

    # 自动检测 bilinear
    detected_bilinear = 'up1.weight' not in state_dict
    if detected_bilinear != args.bilinear:
        print(f'⚠ 警告: 自动检测 bilinear={detected_bilinear}，但参数指定 bilinear={args.bilinear}')
        print(f'  以自动检测结果为准 (bilinear={detected_bilinear})')
        args.bilinear = detected_bilinear
    print(f'bilinear = {args.bilinear}')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net.load_state_dict(state_dict)
    net = net.to(memory_format=torch.channels_last)
    net = net.to(device)
    net.eval()
    torch.backends.cudnn.benchmark = True
    print('✓ 模型加载成功')

    # ---- 选择数据源 ----
    if args.test_zip:
        print(f'测试数据 (zip): {args.test_zip}')
        image_iter = load_test_images_from_zip(args.test_zip)
    elif args.test_dir:
        print(f'测试数据 (dir): {args.test_dir}')
        image_iter = load_test_images_from_dir(args.test_dir)
    else:
        raise ValueError('请指定 --test-dir 或 --test-zip')

    # ---- 批量推理 ----
    print(f'batch_size={args.batch_size} | scale={args.scale} | amp={args.amp}')
    results = []
    batch_tensors = []
    batch_names = []
    batch_hw = None

    with torch.inference_mode():
        for name, img in tqdm(image_iter, desc='预测中'):
            orig_w, orig_h = img.size
            x = torch.from_numpy(
                BasicDataset.preprocess(None, img, args.scale, is_mask=False)
            )

            batch_tensors.append(x)
            batch_names.append(name)
            batch_hw = (orig_h, orig_w)

            if len(batch_tensors) == args.batch_size:
                masks = _run_batch(net, batch_tensors, batch_hw, device, args.amp)
                for i, n in enumerate(batch_names):
                    results.append({'img': n, 'rle_mask': rle_encode(masks[i])})
                batch_tensors.clear()
                batch_names.clear()

        # 最后不足一个 batch
        if batch_tensors:
            masks = _run_batch(net, batch_tensors, batch_hw, device, args.amp)
            for i, n in enumerate(batch_names):
                results.append({'img': n, 'rle_mask': rle_encode(masks[i])})

    # ---- 写 CSV ----
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f'\n✓ {args.output} 已生成')
    print(f'  总行数: {len(df)}')
    print(f'  文件大小: {os.path.getsize(args.output) / 1024**2:.1f} MB')
    print(f'\n前 5 行预览:')
    print(df.head())


def _run_batch(net, batch_tensors, target_hw, device, use_amp):
    batch = torch.stack(batch_tensors).to(
        device, dtype=torch.float32, memory_format=torch.channels_last)
    with torch.autocast('cuda', enabled=use_amp):
        output = net(batch).float()
    output = F.interpolate(output, target_hw, mode='bilinear', align_corners=False)
    return output.argmax(dim=1).cpu().numpy().astype(np.uint8)


if __name__ == '__main__':
    main()
