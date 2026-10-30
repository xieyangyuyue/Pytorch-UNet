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
# 导入命令行参数解析
import argparse
# 导入 IO 模块，用于 BytesIO 读取 zip 内文件
import io
# 导入操作系统接口
import os
# 导入 zip 文件处理
import zipfile
# 导入路径处理
from pathlib import Path

# 导入 numpy
import numpy as np
# 导入 pandas，用于生成提交 CSV
import pandas as pd
# 导入 PyTorch
import torch
# 导入函数式接口，用于 interpolate
import torch.nn.functional as F
# 导入 PIL 图像库
from PIL import Image
# 导入进度条
from tqdm import tqdm

# 导入 U-Net 模型
from unet import UNet
# 导入数据集类（用于预处理）
from utils.data_loading import BasicDataset


def rle_encode(mask, order='F'):
    """Carvana 官方要求列优先 (Fortran order) RLE 编码"""
    # ⚠ 注意：order='F' 是列优先（Fortran 序），这是本脚本的潜在问题
    # predict.py（batch_predict_v2）通过自检验证发现 order='C'（行优先）才是正确的
    # 将二维掩码展平为一维像素序列
    pixels = mask.flatten(order=order)
    # 两端补 0，便于检测跳变位置
    pixels = np.concatenate([[0], pixels, [0]])
    # 找到所有跳变位置（0→1 或 1→0），+1 转为 1-based 索引
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # 将偶数位（结束位置）转为长度：长度 = 结束 - 开始
    runs[1::2] -= runs[::2]
    # 将所有 run 拼成 "start1 length1 start2 length2 ..." 格式的字符串
    return ' '.join(str(x) for x in runs)


def get_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Batch predict masks → submission.csv')
    parser.add_argument('--model', '-m', required=True, help='Path to .pth checkpoint')  # 模型权重路径（必填）
    parser.add_argument('--test-dir', '-d', help='Directory of test images')  # 测试图目录
    parser.add_argument('--test-zip', '-z', help='Zip file of test images')  # 测试图 zip 文件
    parser.add_argument('--output', '-o', default='submission.csv', help='Output CSV path')  # 输出 CSV 路径
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size')  # 批次大小，默认 32
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor (must match training)')  # 缩放因子，必须与训练一致
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')  # 类别数，默认 2
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling (must match training)')  # 双线性上采样
    # ⚠ Bug: action='store_true' + default=True 导致 --amp 永远为 True，无法关闭
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision for inference')  # 混合精度（有 bug）
    return parser.parse_args()  # 解析并返回


def load_test_images_from_dir(test_dir):
    """从目录读图片，返回 (文件名, PIL.Image) 迭代器"""
    # 支持的图像扩展名集合
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    # 列出目录中所有图像文件（排除隐藏文件），按名称排序
    files = sorted(
        f for f in os.listdir(test_dir)
        if Path(f).suffix.lower() in exts and not f.startswith('.')
    )
    # 逐个打开图像，转为 RGB 模式
    for f in files:
        img = Image.open(os.path.join(test_dir, f)).convert('RGB')  # 打开并转 RGB
        yield f, img  # 生成器返回 (文件名, PIL 图像)


def load_test_images_from_zip(zip_path):
    """从 zip 读图片，返回 (文件名, PIL.Image) 迭代器"""
    with zipfile.ZipFile(zip_path, 'r') as zf:  # 打开 zip 文件
        # 列出 zip 中所有图像文件（排除 macOS 的 __MACOSX 目录）
        img_files = sorted([
            f for f in zf.namelist()
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            and not f.startswith('__MACOSX')
        ])
        # 逐个读取图像
        for img_path in img_files:
            with zf.open(img_path) as f:  # 从 zip 中打开文件
                img = Image.open(io.BytesIO(f.read())).convert('RGB')  # 读取到内存并转 RGB
            yield os.path.basename(img_path), img  # 返回 (文件名, PIL 图像)


def main():
    args = get_args()  # 获取命令行参数

    # ---- 设备 ----
    # 选择设备：有 GPU 用 cuda，否则用 cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', end='')  # 打印设备信息
    # if device.type == 'cuda':
    #     # print(f' | {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    # else:
    #     print()

    # ---- 加载权重 ----
    print(f'加载权重: {args.model}')  # 打印权重路径
    # 加载模型权重到 CPU（避免 GPU 内存问题）
    state_dict = torch.load(args.model, map_location=device, weights_only=False)
    state_dict.pop('mask_values', None)  # 移除 mask_values（非模型参数）

    # 自动检测 bilinear：转置卷积版有 up1.weight，双线性版没有
    detected_bilinear = 'up1.weight' not in state_dict
    if detected_bilinear != args.bilinear:  # 如果自动检测与参数指定不一致
        print(f'⚠ 警告: 自动检测 bilinear={detected_bilinear}，但参数指定 bilinear={args.bilinear}')
        print(f'  以自动检测结果为准 (bilinear={detected_bilinear})')
        args.bilinear = detected_bilinear  # 以自动检测为准
    print(f'bilinear = {args.bilinear}')  # 打印最终 bilinear 设置

    # 创建 U-Net 模型
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net.load_state_dict(state_dict)  # 加载权重
    net = net.to(memory_format=torch.channels_last)  # channels_last 内存格式加速
    net = net.to(device)  # 移到目标设备
    net.eval()  # 切换到评估模式
    torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优
    print('✓ 模型加载成功')  # 打印成功信息

    # ---- 选择数据源 ----
    if args.test_zip:  # 如果指定了 zip 文件
        print(f'测试数据 (zip): {args.test_zip}')
        image_iter = load_test_images_from_zip(args.test_zip)  # 从 zip 加载
    elif args.test_dir:  # 如果指定了目录
        print(f'测试数据 (dir): {args.test_dir}')
        image_iter = load_test_images_from_dir(args.test_dir)  # 从目录加载
    else:
        raise ValueError('请指定 --test-dir 或 --test-zip')  # 必须指定数据源

    # ---- 批量推理 ----
    print(f'batch_size={args.batch_size} | scale={args.scale} | amp={args.amp}')  # 打印推理参数
    results = []  # 存储推理结果 (文件名, RLE 编码)
    batch_tensors = []  # 当前 batch 的图像张量列表
    batch_names = []    # 当前 batch 的文件名列表
    batch_hw = None     # 当前 batch 的原始尺寸 (高, 宽)

    with torch.inference_mode():  # 禁用梯度计算
        # 遍历所有测试图像
        for name, img in tqdm(image_iter, desc='预测中'):
            orig_w, orig_h = img.size  # 获取原始宽高
            # 预处理图像：使用 BasicDataset.preprocess（缩放 + CHW + 归一化）
            x = torch.from_numpy(
                BasicDataset.preprocess(None, img, args.scale, is_mask=False)
            )

            batch_tensors.append(x)  # 添加到当前 batch
            batch_names.append(name)  # 记录文件名
            batch_hw = (orig_h, orig_w)  # 记录原始尺寸

            # 当 batch 满了，执行推理
            if len(batch_tensors) == args.batch_size:
                masks = _run_batch(net, batch_tensors, batch_hw, device, args.amp)  # 批量推理
                for i, n in enumerate(batch_names):  # 遍历结果
                    results.append({'img': n, 'rle_mask': rle_encode(masks[i])})  # RLE 编码并保存
                batch_tensors.clear()  # 清空 batch
                batch_names.clear()

        # 最后不足一个 batch 的剩余图像
        if batch_tensors:
            masks = _run_batch(net, batch_tensors, batch_hw, device, args.amp)  # 推理剩余图像
            for i, n in enumerate(batch_names):
                results.append({'img': n, 'rle_mask': rle_encode(masks[i])})

    # ---- 写 CSV ----
    df = pd.DataFrame(results)  # 转为 DataFrame
    df.to_csv(args.output, index=False)  # 写入 CSV（不保存索引列）
    print(f'\n✓ {args.output} 已生成')  # 打印输出路径
    print(f'  总行数: {len(df)}')  # 打印总行数
    print(f'  文件大小: {os.path.getsize(args.output) / 1024**2:.1f} MB')  # 打印文件大小
    print(f'\n前 5 行预览:')  # 打印前 5 行预览
    print(df.head())


def _run_batch(net, batch_tensors, target_hw, device, use_amp):
    # 执行一个 batch 的推理
    # net:           模型
    # batch_tensors: 图像张量列表
    # target_hw:     目标尺寸 (高, 宽)，用于将预测掩码放大回原图尺寸
    # device:        计算设备
    # use_amp:       是否使用混合精度
    # 将列表堆叠为 batch 张量，移到 GPU，float32，channels_last
    batch = torch.stack(batch_tensors).to(
        device, dtype=torch.float32, memory_format=torch.channels_last)
    with torch.autocast('cuda', enabled=use_amp):  # 混合精度上下文
        output = net(batch).float()  # 前向传播，转回 float32
    # 将预测掩码双线性插值放大回原图尺寸
    output = F.interpolate(output, target_hw, mode='bilinear', align_corners=False)
    # argmax 得到类别，转为 numpy uint8
    return output.argmax(dim=1).cpu().numpy().astype(np.uint8)


if __name__ == '__main__':
    main()  # 程序入口
