#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_predict_v2.py —— Carvana 批量推理 + 五级自检

相比 v1 修复的关键问题：
  1. 【致命】RLE 编码顺序：'F' → 'C'（已用官方 train_masks.csv 验证，F 序会导致 Dice ≈ 0.35）
  2. 【致命】img 列保留 .jpg 后缀（官方评估靠它匹配，去掉会 keys not found）
  3. 【致命】全背景掩码返回 '1 0' 而不是空串（空串会被 pandas 读成 NaN，整行作废）
  4. AMP 开关的 argparse bug（store_true + default=True 导致永远关不掉）
  5. 串行解码 → DataLoader 多进程并行解码（这是之前"双卡反而更慢"的真正原因）
  6. 新增断点续跑、小批量试跑、五级自检

用法：
  # 第 1 步：只跑自检，不推理（约 2 分钟，必须先全部 PASS）
  python batch_predict_v2.py -m ckpt.pth -d ./data/test_hq --selftest

  # 第 2 步：小批量试跑 500 张，先提交验分
  python batch_predict_v2.py -m ckpt.pth -d ./data/test_hq --limit 500 -o sub_500.csv

  # 第 3 步：全量推理
  python batch_predict_v2.py -m ckpt.pth -d ./data/test_hq -o submission.csv -b 16 --amp
"""

import argparse
import glob
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from unet import UNet

try:
    from utils.data_loading import BasicDataset
except Exception:                                    # 没这个模块也能跑（自检会提示）
    BasicDataset = None


# ===========================================================================
# 全局常量
# ===========================================================================
RLE_ORDER = 'C'          # ★ 由自检 1 用官方 train_masks.csv 验证得出，勿改
KEEP_EXT  = True         # ★ img 列保留 .jpg，官方评估依赖后缀匹配
EPS       = 1e-7


# ===========================================================================
# 一、RLE 编解码与评估（纯函数，可独立测试）
# ===========================================================================
def rle_encode(mask: np.ndarray, order: str = RLE_ORDER) -> str:
    """二值掩码 → Carvana RLE 字符串。

    order='C' 行优先（已验证与官方 train_masks.csv 完全一致）
    order='F' 列优先（错误，Dice 会掉到 0.35 左右）
    """
    mask = (mask > 0).astype(np.uint8)
    pixels = mask.flatten(order=order)
    pixels = np.concatenate([[0], pixels, [0]])       # 两端补 0，便于定位跳变
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if runs.size == 0:
        return '1 0'                                   # 全背景：返回合法占位，避免 CSV 变 NaN
    runs[1::2] -= runs[::2]                            # 结束位置 → 长度
    return ' '.join(str(int(x)) for x in runs)


def rle_decode(rle_str: str, H: int, W: int) -> np.ndarray:
    """RLE 字符串 → 掩码。模拟 Kaggle 官方评估端的解码行为（C 序 reshape）。"""
    if rle_str is None or (isinstance(rle_str, float) and np.isnan(rle_str)):
        return np.zeros((H, W), np.uint8)
    s = str(rle_str).strip()
    if s == '':
        return np.zeros((H, W), np.uint8)
    a = np.array([int(x) for x in s.split()], dtype=np.int64).reshape(-1, 2)
    flat = np.zeros(H * W, np.uint8)
    for start, length in a:
        lo = int(start) - 1
        hi = min(lo + int(length), H * W)
        if hi > lo:
            flat[lo:hi] = 1
    return flat.reshape(H, W)                          # C 序，与官方一致


def dice_coeff(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice 系数，与 Kaggle 该赛题的评分指标一致。"""
    p = (pred > 0).astype(bool)
    t = (target > 0).astype(bool)
    if p.sum() == 0 and t.sum() == 0:
        return 1.0
    return float(2 * (p & t).sum() / (p.sum() + t.sum() + EPS))


def check_rle_format(rle_str, H: int, W: int):
    """校验单条 RLE 是否满足 Kaggle 要求：sorted / positive / 不重复。

    返回 (是否合法, 原因)。
    """
    if rle_str is None or (isinstance(rle_str, float) and np.isnan(rle_str)):
        return False, 'NaN'
    s = str(rle_str).strip()
    if s == '':
        return False, '空字符串'
    toks = s.split()
    if len(toks) % 2 != 0:
        return False, f'数字个数为奇数({len(toks)})'
    try:
        arr = np.array([int(x) for x in toks], dtype=np.int64).reshape(-1, 2)
    except Exception:
        return False, '含非整数'
    starts, lengths = arr[:, 0], arr[:, 1]
    if (starts < 1).any():
        return False, 'start 必须 >= 1'
    if (lengths < 0).any():
        return False, 'length 必须 >= 0'
    if starts.size > 1 and not (np.diff(starts) > 0).all():
        return False, 'start 未严格递增'
    ends = starts + lengths
    if starts.size > 1 and not (ends[:-1] <= starts[1:]).all():
        return False, '区间重叠'
    if ends[-1] - 1 > H * W:
        return False, f'越界(最大索引 {ends[-1]-1} > {H*W})'
    return True, 'ok'


# ===========================================================================
# 二、预处理（与训练保持一致）
# ===========================================================================
def preprocess(pil_img: Image.Image, scale: float, is_mask: bool = False) -> np.ndarray:
    """等价于 BasicDataset.preprocess：resize → HWC2CHW → /255。"""
    w, h = pil_img.size
    new_w, new_h = int(scale * w), int(scale * h)
    assert new_w > 0 and new_h > 0, f'scale={scale} 太小，缩放后无像素'
    pil_img = pil_img.resize((new_w, new_h),
                             resample=Image.NEAREST if is_mask else Image.BICUBIC)
    a = np.asarray(pil_img)
    if not is_mask:
        if a.ndim == 2:
            a = a[np.newaxis, ...]
        else:
            a = a.transpose((2, 0, 1))
        a = a / 255.0
    return np.ascontiguousarray(a, dtype=np.float32)


# ===========================================================================
# 三、模型加载（自动适配单卡 / 多卡）
# ===========================================================================
def load_model(ckpt_path: str, n_classes: int = 2, device: torch.device = None):
    """载入权重，自动推断 bilinear，自动剥离 module. 前缀，自动决定是否 DataParallel。"""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd.pop('mask_values', None)                        # 数据集信息，不是权重

    # 自动判断 bilinear：转置卷积版才有 up1.weight
    bilinear = 'up1.weight' not in sd
    print(f'[模型] 自动检测 bilinear = {bilinear}（必须和训练一致）')

    # 多卡存的权重带 module. 前缀，先剥掉再喂给裸模型
    if any(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}

    net = UNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)
    net.load_state_dict(sd)                            # strict=True：结构不对会立刻报错
    print('[模型] 权重加载成功')

    net = net.to(memory_format=torch.channels_last)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(n_gpu)), output_device=0)
        print(f'[模型] 已启用 DataParallel，使用 {n_gpu} 张 GPU')
    else:
        print(f'[模型] 单卡模式（检测到 {n_gpu} 张 GPU）')

    net = net.to(device)
    net.eval()
    return net, bilinear


# ===========================================================================
# 四、数据管道（多进程并行解码 —— 这是提速的关键）
# ===========================================================================
class TestDataset(Dataset):
    def __init__(self, paths, scale):
        self.paths = paths
        self.scale = scale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert('RGB')
        w, h = img.size
        a = preprocess(img, self.scale)
        name = os.path.basename(p) if KEEP_EXT else os.path.basename(p)[:-4]
        return torch.from_numpy(a), name, h, w        # 返回原图尺寸，用于还原


def collate(batch):
    xs   = torch.stack([b[0] for b in batch])
    name = [b[1] for b in batch]
    hs   = {b[2] for b in batch}
    ws   = {b[3] for b in batch}
    assert len(hs) == 1 and len(ws) == 1, 'batch 内图片尺寸不一致，请逐张处理'
    return xs, name, hs.pop(), ws.pop()


def build_loader(paths, scale, batch_size, workers):
    return DataLoader(
        TestDataset(paths, scale),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        collate_fn=collate, drop_last=False,
    )


def collect_paths(test_dir=None, test_zip=None, out_root='/kaggle/tmp/test_images'):
    """返回 (图片绝对路径列表, 是否需要清理)。支持目录和 zip（zip 会预解压到临时盘）。"""
    if test_dir:
        paths = sorted(glob.glob(os.path.join(test_dir, '**', '*.jpg'), recursive=True))
        return paths, False
    if test_zip:
        os.makedirs(out_root, exist_ok=True)
        if len(glob.glob(os.path.join(out_root, '**', '*.jpg'), recursive=True)) < 1000:
            print(f'[数据] 解压 {test_zip} → {out_root}（15GB 约需 3-8 分钟）')
            t = time.time()
            r = subprocess.run(['unzip', '-q', '-o', test_zip, '-d', out_root],
                               capture_output=True, text=True)
            print(f'[数据] 解压完成，用时 {(time.time()-t)/60:.1f} min')
            if r.returncode != 0:
                print('[数据] !! unzip 报错:\n', r.stderr[-2000:])
        paths = sorted(glob.glob(os.path.join(out_root, '**', '*.jpg'), recursive=True))
        return paths, False
    raise ValueError('请指定 --test-dir 或 --test-zip')


# ===========================================================================
# 五、推理主循环
# ===========================================================================
def run_predict(net, loader, device, use_amp, output, total_expected,
                save_every=10000, part_path='/kaggle/tmp/submission_part.csv'):
    results = []
    t0 = time.time()
    os.makedirs(os.path.dirname(part_path) or '.', exist_ok=True)

    with torch.inference_mode():
        for bi, (x, names, oh, ow) in enumerate(tqdm(loader, desc='预测中')):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast('cuda', enabled=use_amp and torch.cuda.is_available()):
                out = net(x).float()

            # 先在小图上 argmax，再 NEAREST 放大：省 30 倍带宽，边界更锐利
            m = out.argmax(dim=1).unsqueeze(1).float()
            m = F.interpolate(m, size=(oh, ow), mode='nearest')
            m = m[:, 0].byte().cpu().numpy()

            for j, name in enumerate(names):
                results.append((name, rle_encode(m[j])))

            if save_every and (bi + 1) % max(1, save_every // loader.batch_size) == 0:
                pd.DataFrame(results, columns=['img', 'rle_mask']).to_csv(part_path, index=False)
                el = time.time() - t0
                n = len(results)
                exp = total_expected or n
                print(f'  [{n}/{exp}] 已用 {el/60:.1f}min, '
                      f'预计剩余 {el/max(n,1)*(exp-n)/60:.1f}min')

    df = pd.DataFrame(results, columns=['img', 'rle_mask'])
    df.to_csv(output, index=False)
    print(f'\n[推理] 完成：{len(df)} 行, 耗时 {(time.time()-t0)/3600:.2f}h → {output}')
    return df


# ===========================================================================
# 六、五级自检
# ===========================================================================
def _tag(ok):
    return '✓ PASS' if ok else '✗ FAIL'


def selftest_0_rle_roundtrip():
    """自检 0：RLE 编解码往返一致性（纯函数，不需要 GPU/数据，秒级）

    查什么：encode → decode 能否逐像素还原；边界情况（全 0 / 全 1 / 单像素 / 首行末行）。
    判定   ：全部还原一致且格式合法。
    失败   ：rle_encode / rle_decode 实现有误，后面全不用跑。
    """
    print('\n' + '=' * 70)
    print('自检 0：RLE 编解码往返一致性')
    print('=' * 70)
    H, W = 64, 100
    rng = np.random.default_rng(0)
    cases = {
        '随机噪点':      (rng.random((H, W)) > 0.7).astype(np.uint8),
        '块状区域':      np.zeros((H, W), np.uint8),
        '全背景(全0)':   np.zeros((H, W), np.uint8),
        '全前景(全1)':   np.ones((H, W), np.uint8),
        '左上角单像素':  np.zeros((H, W), np.uint8),
        '右下角单像素':  np.zeros((H, W), np.uint8),
        '整行':          np.zeros((H, W), np.uint8),
    }
    cases['块状区域'][10:30, 20:60] = 1
    cases['左上角单像素'][0, 0] = 1
    cases['右下角单像素'][H - 1, W - 1] = 1
    cases['整行'][5, :] = 1

    all_ok = True
    for name, m in cases.items():
        s = rle_encode(m)
        ok_fmt, why = check_rle_format(s, H, W)
        back = rle_decode(s, H, W)
        ok_rt = np.array_equal(back, (m > 0).astype(np.uint8))
        ok = ok_fmt and ok_rt
        all_ok &= ok
        print(f'  {_tag(ok)}  {name:<12} 前景={m.mean():.3f}  往返一致={ok_rt}  格式={why}')
        if not ok:
            print(f'        编码片段: {s[:60]}')
    print(f'\n  >>> 自检 0 总体：{_tag(all_ok)}')
    return all_ok


def selftest_1_official_csv(train_csv=None, mask_dir='./data/masks', n=50):
    """自检 1：与官方 train_masks.csv 逐条比对（权威标尺，决定 RLE_ORDER）

    查什么：用自己的 gif 掩码重新编码，能否复现官方 RLE 字符串；img 列是否带 .jpg。
    判定   ：C 序（或 F 序）匹配率 100%。
    失败   ：说明编码顺序或 reshape 方式与官方不一致，提交必错。
    """
    print('\n' + '=' * 70)
    print('自检 1：与官方 train_masks.csv 比对（决定编码顺序）')
    print('=' * 70)

    # 自动定位官方 CSV（支持 zip）
    if train_csv is None:
        for p in ['/kaggle/input/competitions/carvana-image-masking-challenge/train_masks.csv.zip',
                  './data/train_masks.csv', './train_masks.csv']:
            if os.path.exists(p):
                train_csv = p
                break
    if train_csv is None or not os.path.exists(train_csv):
        print('  ⚠ 找不到官方 train_masks.csv，跳过（强烈建议补上，它是最权威的标尺）')
        return None
    if train_csv.endswith('.zip'):
        tmp = '/kaggle/tmp/official'
        os.makedirs(tmp, exist_ok=True)
        with zipfile.ZipFile(train_csv) as z:
            z.extractall(tmp)
        train_csv = os.path.join(tmp, 'train_masks.csv')

    df = pd.read_csv(train_csv)
    print(f'  官方 CSV: {len(df)} 行 | 前 3 个 img: {list(df["img"][:3])}')
    has_ext = str(df['img'].iloc[0]).endswith('.jpg')
    print(f'  {_tag(has_ext)} 官方 img 列{"带" if has_ext else "不带"} .jpg 后缀'
          f'  → 代码 KEEP_EXT 应为 {has_ext}')

    ok_c = ok_f = 0
    checked = 0
    for i in range(min(n, len(df))):
        r = df.iloc[i]
        raw = str(r['img'])
        iid = raw[:-4] if raw.endswith('.jpg') else raw
        mp = os.path.join(mask_dir, f'{iid}_mask.gif')
        if not os.path.exists(mp):
            continue
        m = (np.array(Image.open(mp)) > 0).astype(np.uint8)
        official = str(r['rle_mask'])
        ok_c += (rle_encode(m, 'C') == official)
        ok_f += (rle_encode(m, 'F') == official)
        checked += 1

    if checked == 0:
        print('  ⚠ 没找到对应的 gif 掩码，无法比对。检查 --mask-dir')
        return None

    print(f'  比对 {checked} 条：')
    print(f'    C 序(行优先) 匹配 {ok_c}/{checked}  {_tag(ok_c == checked)}')
    print(f'    F 序(列优先) 匹配 {ok_f}/{checked}  {_tag(ok_f == checked)}')
    verdict = 'C' if ok_c == checked else ('F' if ok_f == checked else None)
    print(f'\n  >>> 结论：应使用 order = {verdict}  (当前代码 RLE_ORDER = {RLE_ORDER})')
    if verdict and verdict != RLE_ORDER:
        print(f'  ✗ 严重：代码里的 RLE_ORDER 必须改成 "{verdict}"！')
        return False
    return verdict == RLE_ORDER


def selftest_2_model_e2e(net, device, img_dir='./data/imgs', mask_dir='./data/masks',
                         scale=0.5, n=20, use_amp=False):
    """自检 2：模型端到端 Dice（同时验证模型 / 预处理 / scale / 编码顺序）

    查什么：训练图上跑模型 → 编码 → 解码 → 与真值 gif 比 Dice；C 序和 F 序各算一遍。
    判定   ：C 序 Dice > 0.95，且 C 序明显高于 F 序。
    失败   ：C 序也不高 → 模型或 scale 有问题；C/F 差不多 → 掩码可能是特殊形状。
    """
    print('\n' + '=' * 70)
    print('自检 2：模型端到端 Dice（在训练集上闭环验证）')
    print('=' * 70)
    paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))[:n]
    if not paths:
        print(f'  ⚠ {img_dir} 下没有图片，跳过')
        return None

    dice_c, dice_f, ratio = [], [], []
    for p in paths:
        iid = os.path.basename(p)[:-4]
        mp = os.path.join(mask_dir, f'{iid}_mask.gif')
        if not os.path.exists(mp):
            continue
        img = Image.open(p).convert('RGB')
        ow, oh = img.size
        x = torch.from_numpy(preprocess(img, scale))[None].to(device)
        with torch.inference_mode():
            out = net(x).float()
        out = F.interpolate(out, size=(oh, ow), mode='nearest')[0]
        pred = out.argmax(0).cpu().numpy().astype(np.uint8)

        gt = (np.array(Image.open(mp)) > 0).astype(np.uint8)
        dc = dice_coeff(rle_decode(rle_encode(pred, 'C'), oh, ow), gt)
        df_ = dice_coeff(rle_decode(rle_encode(pred, 'F'), oh, ow), gt)
        dice_c.append(dc)
        dice_f.append(df_)
        ratio.append(pred.mean())

    if not dice_c:
        print('  ⚠ 没有匹配到 gif 掩码，跳过')
        return None

    mc, mf = np.mean(dice_c), np.mean(dice_f)
    print(f'  样本数 {len(dice_c)} | 预测前景占比均值 {np.mean(ratio):.3f}')
    print(f'    C 序 Dice : {mc:.5f}   {_tag(mc > 0.95)}')
    print(f'    F 序 Dice : {mf:.5f}   {_tag(mf > 0.95)}')
    print(f'    C 序高出  : {(mc - mf):.5f}')
    print(f'\n  >>> 结论：{"C 序正确，模型与编码都正常" if mc > 0.95 and mc > mf else "异常，见上方数值"}')
    if mc > 0.95 and mc > mf:
        print(f'  ✓ 预期提交分数应在 {mc:.3f} 附近（测试集会略低于训练图）')
        return True
    return False


def selftest_3_format(df, expected_ids=None, H=1280, W=1918, n_check=3000):
    """自检 3：提交文件格式终检（提交前的最后一道闸）

    查什么：行数、img 后缀、img 集合匹配、NaN、RLE 合法性、前景占比、全背景数量。
    判定   ：全部通过。
    失败   ：任何一个 FAIL 都会导致 0 分或大幅扣分，必须修。
    """
    print('\n' + '=' * 70)
    print('自检 3：提交文件格式终检')
    print('=' * 70)
    ok_all = True

    # 1) 行数
    n = len(df)
    exp = len(expected_ids) if expected_ids is not None else 100064
    ok = (n == exp)
    ok_all &= ok
    print(f'  {_tag(ok)} 行数 {n}（期望 {exp}）')
    if not ok:
        print(f'        缺 {exp-n} 行会被记 0 分！总分约被拉低到 {n/exp:.4f}')

    # 2) img 列后缀
    has_ext = df['img'].astype(str).str.endswith('.jpg').all()
    ok_all &= bool(has_ext)
    print(f'  {_tag(has_ext)} img 列全部带 .jpg 后缀（官方依赖后缀匹配）')

    # 3) img 集合是否与官方一致
    if expected_ids is not None:
        miss = set(expected_ids) - set(df['img'].astype(str))
        extra = set(df['img'].astype(str)) - set(expected_ids)
        ok = (len(miss) == 0 and len(extra) == 0)
        ok_all &= ok
        print(f'  {_tag(ok)} img 集合匹配（缺 {len(miss)} / 多 {len(extra)}）')
        if len(miss):
            print(f'        缺失样例: {list(miss)[:5]}')

    # 4) NaN / 空值
    nan_n = df['rle_mask'].isna().sum()
    ok = (nan_n == 0)
    ok_all &= ok
    print(f'  {_tag(ok)} rle_mask 无 NaN（当前 {nan_n} 个）')

    # 5) RLE 合法性抽样
    bad = []
    sample = df.sample(min(n_check, n), random_state=0) if n else df
    for _, r in sample.iterrows():
        okk, why = check_rle_format(str(r['rle_mask']), H, W)
        if not okk:
            bad.append((r['img'], why))
    ok = (len(bad) == 0)
    ok_all &= ok
    print(f'  {_tag(ok)} RLE 格式合法（抽查 {len(sample)} 条，非法 {len(bad)}）')
    for b in bad[:3]:
        print(f'        {b[0]}: {b[1]}')

    # 6) 前景占比合理性
    means = []
    for _, r in sample.iterrows():
        m = rle_decode(str(r['rle_mask']), H, W)
        means.append(m.mean())
    if means:
        mm = float(np.mean(means))
        ok = (0.05 < mm < 0.60)
        ok_all &= ok
        print(f'  {_tag(ok)} 预测前景占比均值 {mm:.3f}（合理区间 0.05~0.60）')
        if not ok:
            print(f'        ← 若接近 0 说明预测全背景；接近 1 说明全前景，模型或编码有问题')

    # 7) 全背景（'1 0'）数量
    n_empty = (df['rle_mask'].astype(str).str.strip() == '1 0').sum()
    ok = (n_empty < n * 0.05)
    ok_all &= ok
    print(f'  {_tag(ok)} 全背景图片 {n_empty} 张（占比 {n_empty/max(n,1):.2%}，应 < 5%）')
    if n_empty > n * 0.05:
        print(f'        ← 大量全背景说明模型没学会，或对测试图读取失败')

    print(f'\n  >>> 自检 3 总体：{_tag(ok_all)}')
    return bool(ok_all)


def selftest_4_visual(df, img_dir, out_dir='./viz_check', n=4):
    """自检 4：肉眼可视化（可选，最有说服力）

    查什么：把预测的 RLE 解码后叠在原图上，看轮廓是否贴合车身。
    判定   ：肉眼确认掩码覆盖车身。
    失败   ：若掩码呈条纹/斜切/整体错位 → 编码顺序错；若形状对但整体偏移 → resize 有问题。
    """
    print('\n' + '=' * 70)
    print('自检 4：可视化抽查')
    print('=' * 70)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('  ⚠ 没有 matplotlib，跳过可视化')
        return None

    os.makedirs(out_dir, exist_ok=True)
    rows = df.head(n)
    for i, (_, r) in enumerate(rows.iterrows()):
        name = str(r['img'])
        p = os.path.join(img_dir, name if name.endswith('.jpg') else name + '.jpg')
        if not os.path.exists(p):
            alt = glob.glob(os.path.join(img_dir, '**', name + '*.jpg'), recursive=True)
            if not alt:
                continue
            p = alt[0]
        img = Image.open(p).convert('RGB')
        W, H = img.size
        m = rle_decode(str(r['rle_mask']), H, W)

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        ax[0].imshow(img); ax[0].set_title('原图')
        ax[1].imshow(m, cmap='gray'); ax[1].set_title(f'预测掩码 (前景{m.mean():.3f})')
        ax[2].imshow(img)
        ax[2].imshow(np.ma.masked_where(m == 0, m), cmap='autumn', alpha=0.5)
        ax[2].set_title('叠加')
        for a in ax:
            a.axis('off')
        fp = os.path.join(out_dir, f'{i:02d}_{name}.png')
        plt.tight_layout(); plt.savefig(fp, dpi=90); plt.close(fig)
        print(f'  已保存: {fp}')
    print(f'  >>> 请打开 {out_dir}/ 下的图片肉眼确认掩码是否贴合车身')
    return True


# ===========================================================================
# 七、CLI
# ===========================================================================
def get_args():
    ap = argparse.ArgumentParser(description='Carvana 批量推理 + 五级自检',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model', '-m', required=True, help='.pth 权重路径')
    ap.add_argument('--test-dir', '-d', help='测试图目录')
    ap.add_argument('--test-zip', '-z', help='测试图 zip（自动预解压到临时盘）')
    ap.add_argument('--output', '-o', default='submission.csv', help='输出 CSV')
    ap.add_argument('--batch-size', '-b', type=int, default=16, help='batch（多卡为总和）')
    ap.add_argument('--scale', '-s', type=float, default=0.5, help='必须与训练一致')
    ap.add_argument('--classes', '-c', type=int, default=2)
    ap.add_argument('--no-amp', action='store_true', help='关闭混合精度（默认开启）')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 4),
                    help='解码进程数')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 张（试跑用）')
    ap.add_argument('--selftest', action='store_true', help='只跑自检，不推理')
    ap.add_argument('--viz', type=int, default=0, help='可视化前 N 张')
    ap.add_argument('--mask-dir', default='./data/masks', help='训练掩码目录（自检用）')
    ap.add_argument('--train-img-dir', default='./data/imgs', help='训练图目录（自检用）')
    ap.add_argument('--official-csv', default=None, help='官方 train_masks.csv(.zip)')
    ap.add_argument('--sample-sub', default=None, help='官方 sample_submission.csv（校验 img 集合）')
    return ap.parse_args()


def main():
    args = get_args()
    use_amp = not args.no_amp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[环境] device={device} | GPU={torch.cuda.device_count()} | amp={use_amp} | scale={args.scale}')

    # ---------- 自检 0：纯函数，永远先跑 ----------
    ok0 = selftest_0_rle_roundtrip()

    net, bilinear = load_model(args.model, args.classes, device)

    # ---------- 自检 1 / 2 ----------
    ok1 = selftest_1_official_csv(args.official_csv, args.mask_dir)
    ok2 = selftest_2_model_e2e(net, device, args.train_img_dir, args.mask_dir,
                               args.scale, n=20)

    if not ok0 or ok1 is False or ok2 is False:
        print('\n' + '!' * 70)
        print('! 自检未通过，禁止全量推理 —— 先修上面的 FAIL，否则 10 万张白跑')
        print('!' * 70)
        if args.selftest:
            return
        if input('\n仍要继续？(y/N): ').strip().lower() != 'y':
            return

    if args.selftest:
        print('\n自检模式结束（未执行推理）')
        return

    # ---------- 推理 ----------
    paths, _ = collect_paths(args.test_dir, args.test_zip)
    print(f'[数据] 找到 {len(paths)} 张测试图')
    assert len(paths) > 0, '没找到测试图，检查 --test-dir / --test-zip'
    if args.limit:
        paths = paths[:args.limit]
        print(f'[数据] 试跑模式：只处理前 {len(paths)} 张')

    # 官方 sample_submission 用于校验 img 集合
    expected_ids = None
    if args.sample_sub and os.path.exists(args.sample_sub):
        expected_ids = pd.read_csv(args.sample_sub)['img'].astype(str).tolist()
        print(f'[数据] 官方期望 {len(expected_ids)} 行')

    loader = build_loader(paths, args.scale, args.batch_size, args.workers)
    df = run_predict(net, loader, device, use_amp, args.output,
                     total_expected=len(expected_ids) if expected_ids else len(paths))

    # ---------- 自检 3 / 4 ----------
    ok3 = selftest_3_format(df, expected_ids)
    if args.viz:
        selftest_4_visual(df, args.test_dir or '/kaggle/tmp/test_images', n=args.viz)

    print('\n' + '=' * 70)
    if ok3:
        print(f'✓ 全部通过，可以提交：{args.output}')
    else:
        print(f'✗ 格式有问题，修完再提交（见自检 3）')
    print('=' * 70)


if __name__ == '__main__':
    main()
