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

import argparse  # 命令行参数解析
import glob      # 文件模式匹配
import io        # IO 操作（BytesIO 等）
import json      # JSON 处理（本文件未直接使用）
import os        # 操作系统接口
import subprocess  # 子进程调用（用于 unzip）
import sys       # 系统模块（本文件未直接使用）
import time      # 时间计时
import zipfile   # zip 文件处理
from pathlib import Path  # 路径处理

import numpy as np       # 数值计算
import pandas as pd      # 数据表处理（CSV 读写）
import torch             # PyTorch 深度学习框架
import torch.nn.functional as F  # 函数式接口（interpolate 等）
from PIL import Image    # 图像处理
from torch.utils.data import Dataset, DataLoader  # 数据集与加载器
from tqdm import tqdm    # 进度条

from unet import UNet    # U-Net 模型

try:
    from utils.data_loading import BasicDataset  # 尝试导入预处理工具
except Exception:                                    # 没这个模块也能跑（自检会提示）
    BasicDataset = None


# ===========================================================================
# 全局常量
# ===========================================================================
RLE_ORDER = 'C'          # ★ RLE 编码顺序：'C'=行优先（已验证与官方一致），'F'=列优先（错误）
KEEP_EXT  = True         # ★ img 列保留 .jpg 后缀（官方评估依赖后缀匹配）
EPS       = 1e-7         # 极小常数，防止 Dice 计算除零


# ===========================================================================
# 一、RLE 编解码与评估（纯函数，可独立测试）
# ===========================================================================
def rle_encode(mask: np.ndarray, order: str = RLE_ORDER) -> str:
    """二值掩码 → Carvana RLE 字符串。order='C' 行优先（正确），order='F' 列优先（错误）"""
    mask = (mask > 0).astype(np.uint8)               # 二值化：>0 的像素设为 1
    pixels = mask.flatten(order=order)                # 按指定顺序展平为 1D
    pixels = np.concatenate([[0], pixels, [0]])       # 两端补 0，便于定位跳变
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # 找跳变位置，+1 转 1-based
    if runs.size == 0:
        return '1 0'                                   # 全背景：返回合法占位，避免 NaN
    runs[1::2] -= runs[::2]                            # 结束位置 → 长度
    return ' '.join(str(int(x)) for x in runs)        # 拼成 "start len start len ..."


def rle_decode(rle_str: str, H: int, W: int) -> np.ndarray:
    """RLE 字符串 → 掩码。模拟 Kaggle 官方评估端的解码行为（C 序 reshape）。"""
    if rle_str is None or (isinstance(rle_str, float) and np.isnan(rle_str)):
        return np.zeros((H, W), np.uint8)             # 空值返回全零掩码
    s = str(rle_str).strip()                          # 转字符串去空白
    if s == '':
        return np.zeros((H, W), np.uint8)             # 空串返回全零
    a = np.array([int(x) for x in s.split()], dtype=np.int64).reshape(-1, 2)  # 解析 (start, length) 对
    flat = np.zeros(H * W, np.uint8)                  # 创建 1D 全零数组
    for start, length in a:                           # 遍历每个 run
        lo = int(start) - 1                           # 转 0-based 索引
        hi = min(lo + int(length), H * W)             # 计算结束位置（防越界）
        if hi > lo:
            flat[lo:hi] = 1                           # 设置前景像素
    return flat.reshape(H, W)                          # C 序 reshape，与官方一致


def dice_coeff(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice 系数，与 Kaggle 该赛题的评分指标一致。"""
    p = (pred > 0).astype(bool)                       # 二值化预测
    t = (target > 0).astype(bool)                     # 二值化真实
    if p.sum() == 0 and t.sum() == 0:
        return 1.0                                    # 两者都空 → 完全一致
    return float(2 * (p & t).sum() / (p.sum() + t.sum() + EPS))  # Dice = 2|A∩B|/(|A|+|B|)


def check_rle_format(rle_str, H: int, W: int):
    """校验单条 RLE 是否满足 Kaggle 要求：sorted / positive / 不重复。返回 (是否合法, 原因)。"""
    if rle_str is None or (isinstance(rle_str, float) and np.isnan(rle_str)):
        return False, 'NaN'                           # NaN 不合法
    s = str(rle_str).strip()                          # 转字符串
    if s == '':
        return False, '空字符串'                      # 空串不合法
    toks = s.split()                                  # 按空格分割
    if len(toks) % 2 != 0:
        return False, f'数字个数为奇数({len(toks)})'  # 必须偶数个（start/length 对）
    try:
        arr = np.array([int(x) for x in toks], dtype=np.int64).reshape(-1, 2)  # 解析为整数对
    except Exception:
        return False, '含非整数'                      # 非数字字符
    starts, lengths = arr[:, 0], arr[:, 1]            # 拆分起始和长度
    if (starts < 1).any():
        return False, 'start 必须 >= 1'              # 1-based 索引
    if (lengths < 0).any():
        return False, 'length 必须 >= 0'             # 长度非负
    if starts.size > 1 and not (np.diff(starts) > 0).all():
        return False, 'start 未严格递增'             # 起始位置递增
    ends = starts + lengths                           # 计算结束位置
    if starts.size > 1 and not (ends[:-1] <= starts[1:]).all():
        return False, '区间重叠'                     # run 不重叠
    if ends[-1] - 1 > H * W:
        return False, f'越界(最大索引 {ends[-1]-1} > {H*W})'  # 不超总像素数
    return True, 'ok'                                 # 全部通过


# ===========================================================================
# 二、预处理（与训练保持一致）
# ===========================================================================
def preprocess(pil_img: Image.Image, scale: float, is_mask: bool = False) -> np.ndarray:
    """等价于 BasicDataset.preprocess：resize → HWC2CHW → /255。"""
    w, h = pil_img.size                               # 原始宽高
    new_w, new_h = int(scale * w), int(scale * h)     # 缩放后尺寸
    assert new_w > 0 and new_h > 0, f'scale={scale} 太小，缩放后无像素'  # 确保有像素
    pil_img = pil_img.resize((new_w, new_h),          # 缩放：掩码用最近邻，图像用双三次
                             resample=Image.NEAREST if is_mask else Image.BICUBIC)
    a = np.asarray(pil_img)                           # 转 numpy 数组
    if not is_mask:                                   # 图像（非掩码）
        if a.ndim == 2:                               # 灰度图
            a = a[np.newaxis, ...]                    # 加通道维 (H,W)→(1,H,W)
        else:                                         # RGB 图
            a = a.transpose((2, 0, 1))                # HWC→CHW
        a = a / 255.0                                 # 归一化到 [0,1]
    return np.ascontiguousarray(a, dtype=np.float32)  # 确保内存连续


# ===========================================================================
# 三、模型加载（自动适配单卡 / 多卡）
# ===========================================================================
def load_model(ckpt_path: str, n_classes: int = 2, device: torch.device = None):
    """载入权重，自动推断 bilinear，自动剥离 module. 前缀，自动决定是否 DataParallel。"""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # 加载到 CPU
    sd.pop('mask_values', None)                        # 移除非权重字段

    bilinear = 'up1.weight' not in sd                  # 自动检测：转置卷积版有 up1.weight
    print(f'[模型] 自动检测 bilinear = {bilinear}（必须和训练一致）')

    if any(k.startswith('module.') for k in sd):       # 多卡权重有 module. 前缀
        sd = {k[len('module.'):]: v for k, v in sd.items()}  # 剥离前缀

    net = UNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)  # 创建模型
    net.load_state_dict(sd)                            # 加载权重（strict=True）
    print('[模型] 权重加载成功')

    net = net.to(memory_format=torch.channels_last)    # channels_last 加速
    n_gpu = torch.cuda.device_count()                  # GPU 数量
    if n_gpu > 1:                                     # 多卡
        net = torch.nn.DataParallel(net, device_ids=list(range(n_gpu)), output_device=0)  # 数据并行
        print(f'[模型] 已启用 DataParallel，使用 {n_gpu} 张 GPU')
    else:                                             # 单卡
        print(f'[模型] 单卡模式（检测到 {n_gpu} 张 GPU）')

    net = net.to(device)                               # 移到目标设备
    net.eval()                                        # 评估模式
    return net, bilinear                              # 返回模型和 bilinear 标志


# ===========================================================================
# 四、数据管道（多进程并行解码 —— 这是提速的关键）
# ===========================================================================
class TestDataset(Dataset):
    """测试数据集：只加载图像（无掩码），支持多进程并行预处理"""
    def __init__(self, paths, scale):
        self.paths = paths                             # 图像路径列表
        self.scale = scale                             # 缩放因子

    def __len__(self):
        return len(self.paths)                         # 数据集大小

    def __getitem__(self, i):
        p = self.paths[i]                              # 第 i 个路径
        img = Image.open(p).convert('RGB')             # 打开转 RGB
        w, h = img.size                               # 原始宽高
        a = preprocess(img, self.scale)                # 预处理
        name = os.path.basename(p) if KEEP_EXT else os.path.basename(p)[:-4]  # 文件名（带/不带后缀）
        return torch.from_numpy(a), name, h, w        # 返回 (张量, 文件名, 原始高, 原始宽)


def collate(batch):
    """自定义 collate：堆叠图像张量，收集文件名，校验尺寸一致"""
    xs   = torch.stack([b[0] for b in batch])         # 堆叠图像
    name = [b[1] for b in batch]                       # 文件名列表
    hs   = {b[2] for b in batch}                       # 高度集合（去重）
    ws   = {b[3] for b in batch}                       # 宽度集合（去重）
    assert len(hs) == 1 and len(ws) == 1, 'batch 内图片尺寸不一致，请逐张处理'  # 校验一致
    return xs, name, hs.pop(), ws.pop()               # 返回 (张量, 文件名, 高, 宽)


def build_loader(paths, scale, batch_size, workers):
    """构建 DataLoader"""
    return DataLoader(
        TestDataset(paths, scale),                    # 测试数据集
        batch_size=batch_size, shuffle=False,         # 不打乱
        num_workers=workers, pin_memory=torch.cuda.is_available(),  # 多进程 + 锁页内存
        persistent_workers=workers > 0,               # 保持工作进程
        prefetch_factor=4 if workers > 0 else None,   # 预取因子
        collate_fn=collate, drop_last=False,          # 自定义 collate，不丢弃尾部
    )


def collect_paths(test_dir=None, test_zip=None, out_root='/kaggle/tmp/test_images'):
    """返回 (图片路径列表, 是否需清理)。支持目录和 zip（zip 自动预解压）。"""
    if test_dir:                                       # 指定了目录
        paths = sorted(glob.glob(os.path.join(test_dir, '**', '*.jpg'), recursive=True))  # 递归查找 jpg
        return paths, False
    if test_zip:                                       # 指定了 zip
        os.makedirs(out_root, exist_ok=True)           # 创建临时目录
        if len(glob.glob(os.path.join(out_root, '**', '*.jpg'), recursive=True)) < 1000:  # 图片不足则解压
            print(f'[数据] 解压 {test_zip} → {out_root}（15GB 约需 3-8 分钟）')
            t = time.time()                            # 计时开始
            r = subprocess.run(['unzip', '-q', '-o', test_zip, '-d', out_root],  # 调用 unzip
                               capture_output=True, text=True)
            print(f'[数据] 解压完成，用时 {(time.time()-t)/60:.1f} min')
            if r.returncode != 0:                      # 解压失败
                print('[数据] !! unzip 报错:\n', r.stderr[-2000:])
        paths = sorted(glob.glob(os.path.join(out_root, '**', '*.jpg'), recursive=True))  # 查找 jpg
        return paths, False
    raise ValueError('请指定 --test-dir 或 --test-zip')  # 必须指定数据源


# ===========================================================================
# 五、推理主循环
# ===========================================================================
def run_predict(net, loader, device, use_amp, output, total_expected,
                save_every=10000, part_path='/kaggle/tmp/submission_part.csv'):
    """推理主循环：遍历 DataLoader，批量预测，RLE 编码，定期保存中间结果"""
    results = []                                       # 结果列表 (文件名, RLE)
    t0 = time.time()                                   # 计时开始
    os.makedirs(os.path.dirname(part_path) or '.', exist_ok=True)  # 确保目录存在

    with torch.inference_mode():                       # 禁用梯度
        for bi, (x, names, oh, ow) in enumerate(tqdm(loader, desc='预测中')):  # 遍历 batch
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)  # 移到 GPU
            with torch.autocast('cuda', enabled=use_amp and torch.cuda.is_available()):  # 混合精度
                out = net(x).float()                   # 前向传播

            # 先在小图上 argmax，再 NEAREST 放大：省 30 倍带宽，边界更锐利
            m = out.argmax(dim=1).unsqueeze(1).float()  # argmax 得类别 → 加通道维
            m = F.interpolate(m, size=(oh, ow), mode='nearest')  # 最近邻放大到原图尺寸
            m = m[:, 0].byte().cpu().numpy()           # 去通道维 → uint8 → 移到 CPU

            for j, name in enumerate(names):           # 遍历 batch 中每张图
                results.append((name, rle_encode(m[j])))  # RLE 编码并保存

            # 定期保存中间结果（断点续跑用）
            if save_every and (bi + 1) % max(1, save_every // loader.batch_size) == 0:
                pd.DataFrame(results, columns=['img', 'rle_mask']).to_csv(part_path, index=False)  # 保存中间 CSV
                el = time.time() - t0                  # 已用时间
                n = len(results)                       # 已处理数量
                exp = total_expected or n              # 预期总数
                print(f'  [{n}/{exp}] 已用 {el/60:.1f}min, '
                      f'预计剩余 {el/max(n,1)*(exp-n)/60:.1f}min')  # 打印进度

    df = pd.DataFrame(results, columns=['img', 'rle_mask'])  # 转为 DataFrame
    df.to_csv(output, index=False)                     # 保存最终 CSV
    print(f'\n[推理] 完成：{len(df)} 行, 耗时 {(time.time()-t0)/3600:.2f}h → {output}')  # 打印结果
    return df                                          # 返回 DataFrame


# ===========================================================================
# 六、五级自检
# ===========================================================================
def _tag(ok):
    """返回 PASS/FAIL 标签"""
    return '✓ PASS' if ok else '✗ FAIL'


def selftest_0_rle_roundtrip():
    """自检 0：RLE 编解码往返一致性（纯函数，不需要 GPU/数据，秒级）"""
    print('\n' + '=' * 70)
    print('自检 0：RLE 编解码往返一致性')
    print('=' * 70)
    H, W = 64, 100                                     # 测试用图像尺寸
    rng = np.random.default_rng(0)                     # 随机数生成器（固定种子）
    cases = {                                          # 各种边界情况
        '随机噪点':      (rng.random((H, W)) > 0.7).astype(np.uint8),  # 随机前景
        '块状区域':      np.zeros((H, W), np.uint8),   # 稍后填充
        '全背景(全0)':   np.zeros((H, W), np.uint8),   # 全零
        '全前景(全1)':   np.ones((H, W), np.uint8),    # 全一
        '左上角单像素':  np.zeros((H, W), np.uint8),   # 稍后填充
        '右下角单像素':  np.zeros((H, W), np.uint8),   # 稍后填充
        '整行':          np.zeros((H, W), np.uint8),   # 稍后填充
    }
    cases['块状区域'][10:30, 20:60] = 1                # 填充块状区域
    cases['左上角单像素'][0, 0] = 1                    # 左上角单像素
    cases['右下角单像素'][H - 1, W - 1] = 1            # 右下角单像素
    cases['整行'][5, :] = 1                            # 整行前景

    all_ok = True                                      # 总体结果
    for name, m in cases.items():                      # 遍历每种情况
        s = rle_encode(m)                              # 编码
        ok_fmt, why = check_rle_format(s, H, W)        # 检查格式
        back = rle_decode(s, H, W)                     # 解码
        ok_rt = np.array_equal(back, (m > 0).astype(np.uint8))  # 往返一致性
        ok = ok_fmt and ok_rt                          # 综合判定
        all_ok &= ok                                   # 更新总体
        print(f'  {_tag(ok)}  {name:<12} 前景={m.mean():.3f}  往返一致={ok_rt}  格式={why}')
        if not ok:                                     # 失败时打印编码片段
            print(f'        编码片段: {s[:60]}')
    print(f'\n  >>> 自检 0 总体：{_tag(all_ok)}')
    return all_ok                                      # 返回总体结果


def selftest_1_official_csv(train_csv=None, mask_dir='./data/masks', n=50):
    """自检 1：与官方 train_masks.csv 逐条比对（权威标尺，决定 RLE_ORDER）"""
    print('\n' + '=' * 70)
    print('自检 1：与官方 train_masks.csv 比对（决定编码顺序）')
    print('=' * 70)

    # 自动定位官方 CSV（支持 zip 格式）
    if train_csv is None:                              # 未指定路径
        for p in ['/kaggle/input/competitions/carvana-image-masking-challenge/train_masks.csv.zip',
                  './data/train_masks.csv', './train_masks.csv']:  # 候选路径
            if os.path.exists(p):
                train_csv = p                          # 找到则使用
                break
    if train_csv is None or not os.path.exists(train_csv):  # 找不到
        print('  ⚠ 找不到官方 train_masks.csv，跳过（强烈建议补上，它是最权威的标尺）')
        return None                                    # 跳过
    if train_csv.endswith('.zip'):                     # 如果是 zip
        tmp = '/kaggle/tmp/official'                   # 临时目录
        os.makedirs(tmp, exist_ok=True)
        with zipfile.ZipFile(train_csv) as z:          # 解压
            z.extractall(tmp)
        train_csv = os.path.join(tmp, 'train_masks.csv')  # 指向解压后的 CSV

    df = pd.read_csv(train_csv)                        # 读取官方 CSV
    print(f'  官方 CSV: {len(df)} 行 | 前 3 个 img: {list(df["img"][:3])}')
    has_ext = str(df['img'].iloc[0]).endswith('.jpg')  # 检查 img 列是否带后缀
    print(f'  {_tag(has_ext)} 官方 img 列{"带" if has_ext else "不带"} .jpg 后缀'
          f'  → 代码 KEEP_EXT 应为 {has_ext}')

    ok_c = ok_f = 0                                    # C 序和 F 序匹配计数
    checked = 0                                        # 已检查数量
    for i in range(min(n, len(df))):                   # 遍历前 n 条
        r = df.iloc[i]                                 # 取一行
        raw = str(r['img'])                            # img 列值
        iid = raw[:-4] if raw.endswith('.jpg') else raw  # 去掉 .jpg 后缀
        mp = os.path.join(mask_dir, f'{iid}_mask.gif')  # 掩码文件路径
        if not os.path.exists(mp):                     # 掩码不存在
            continue
        m = (np.array(Image.open(mp)) > 0).astype(np.uint8)  # 读取掩码并二值化
        official = str(r['rle_mask'])                  # 官方 RLE 字符串
        ok_c += (rle_encode(m, 'C') == official)       # C 序匹配
        ok_f += (rle_encode(m, 'F') == official)       # F 序匹配
        checked += 1                                   # 已检查数 +1

    if checked == 0:                                   # 没找到匹配的掩码
        print('  ⚠ 没找到对应的 gif 掩码，无法比对。检查 --mask-dir')
        return None

    print(f'  比对 {checked} 条：')
    print(f'    C 序(行优先) 匹配 {ok_c}/{checked}  {_tag(ok_c == checked)}')
    print(f'    F 序(列优先) 匹配 {ok_f}/{checked}  {_tag(ok_f == checked)}')
    verdict = 'C' if ok_c == checked else ('F' if ok_f == checked else None)  # 判定正确顺序
    print(f'\n  >>> 结论：应使用 order = {verdict}  (当前代码 RLE_ORDER = {RLE_ORDER})')
    if verdict and verdict != RLE_ORDER:               # 如果当前设置错误
        print(f'  ✗ 严重：代码里的 RLE_ORDER 必须改成 "{verdict}"！')
        return False
    return verdict == RLE_ORDER                        # 返回是否正确


def selftest_2_model_e2e(net, device, img_dir='./data/imgs', mask_dir='./data/masks',
                         scale=0.5, n=20, use_amp=False):
    """自检 2：模型端到端 Dice（同时验证模型 / 预处理 / scale / 编码顺序）"""
    print('\n' + '=' * 70)
    print('自检 2：模型端到端 Dice（在训练集上闭环验证）')
    print('=' * 70)
    paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))[:n]  # 取前 n 张训练图
    if not paths:                                      # 没找到图片
        print(f'  ⚠ {img_dir} 下没有图片，跳过')
        return None

    dice_c, dice_f, ratio = [], [], []                 # C序Dice, F序Dice, 前景占比
    for p in paths:                                    # 遍历每张图
        iid = os.path.basename(p)[:-4]                 # 图像 ID（去 .jpg）
        mp = os.path.join(mask_dir, f'{iid}_mask.gif')  # 掩码路径
        if not os.path.exists(mp):                     # 掩码不存在
            continue
        img = Image.open(p).convert('RGB')             # 读取图像
        ow, oh = img.size                              # 原始宽高
        x = torch.from_numpy(preprocess(img, scale))[None].to(device)  # 预处理 → 加 batch 维 → 移到设备
        with torch.inference_mode():                   # 禁用梯度
            out = net(x).float()                       # 前向传播
        out = F.interpolate(out, size=(oh, ow), mode='nearest')[0]  # 放大回原图尺寸 → 去 batch 维
        pred = out.argmax(0).cpu().numpy().astype(np.uint8)  # argmax 得类别 → numpy

        gt = (np.array(Image.open(mp)) > 0).astype(np.uint8)  # 读取真值掩码并二值化
        dc = dice_coeff(rle_decode(rle_encode(pred, 'C'), oh, ow), gt)  # C 序 Dice
        df_ = dice_coeff(rle_decode(rle_encode(pred, 'F'), oh, ow), gt)  # F 序 Dice
        dice_c.append(dc)                              # 记录 C 序 Dice
        dice_f.append(df_)                             # 记录 F 序 Dice
        ratio.append(pred.mean())                      # 记录前景占比

    if not dice_c:                                     # 没有匹配的掩码
        print('  ⚠ 没有匹配到 gif 掩码，跳过')
        return None

    mc, mf = np.mean(dice_c), np.mean(dice_f)          # C 序和 F 序平均 Dice
    print(f'  样本数 {len(dice_c)} | 预测前景占比均值 {np.mean(ratio):.3f}')
    print(f'    C 序 Dice : {mc:.5f}   {_tag(mc > 0.95)}')
    print(f'    F 序 Dice : {mf:.5f}   {_tag(mf > 0.95)}')
    print(f'    C 序高出  : {(mc - mf):.5f}')
    print(f'\n  >>> 结论：{"C 序正确，模型与编码都正常" if mc > 0.95 and mc > mf else "异常，见上方数值"}')
    if mc > 0.95 and mc > mf:                          # C 序正确且优于 F 序
        print(f'  ✓ 预期提交分数应在 {mc:.3f} 附近（测试集会略低于训练图）')
        return True
    return False                                       # 异常


def selftest_3_format(df, expected_ids=None, H=1280, W=1918, n_check=3000):
    """自检 3：提交文件格式终检（提交前的最后一道闸）"""
    print('\n' + '=' * 70)
    print('自检 3：提交文件格式终检')
    print('=' * 70)
    ok_all = True                                      # 总体结果

    # 1) 行数检查
    n = len(df)                                        # 实际行数
    exp = len(expected_ids) if expected_ids is not None else 100064  # 期望行数
    ok = (n == exp)                                    # 行数是否匹配
    ok_all &= ok
    print(f'  {_tag(ok)} 行数 {n}（期望 {exp}）')
    if not ok:                                         # 行数不匹配
        print(f'        缺 {exp-n} 行会被记 0 分！总分约被拉低到 {n/exp:.4f}')

    # 2) img 列后缀检查
    has_ext = df['img'].astype(str).str.endswith('.jpg').all()  # 是否全部带 .jpg
    ok_all &= bool(has_ext)
    print(f'  {_tag(has_ext)} img 列全部带 .jpg 后缀（官方依赖后缀匹配）')

    # 3) img 集合是否与官方一致
    if expected_ids is not None:                       # 有官方 ID 列表
        miss = set(expected_ids) - set(df['img'].astype(str))  # 缺失的 ID
        extra = set(df['img'].astype(str)) - set(expected_ids)  # 多余的 ID
        ok = (len(miss) == 0 and len(extra) == 0)      # 集合完全匹配
        ok_all &= ok
        print(f'  {_tag(ok)} img 集合匹配（缺 {len(miss)} / 多 {len(extra)}）')
        if len(miss):                                  # 有缺失
            print(f'        缺失样例: {list(miss)[:5]}')

    # 4) NaN / 空值检查
    nan_n = df['rle_mask'].isna().sum()                # NaN 数量
    ok = (nan_n == 0)                                  # 无 NaN
    ok_all &= ok
    print(f'  {_tag(ok)} rle_mask 无 NaN（当前 {nan_n} 个）')

    # 5) RLE 合法性抽样检查
    bad = []                                           # 非法记录
    sample = df.sample(min(n_check, n), random_state=0) if n else df  # 随机抽样
    for _, r in sample.iterrows():                     # 遍历抽样
        okk, why = check_rle_format(str(r['rle_mask']), H, W)  # 检查格式
        if not okk:
            bad.append((r['img'], why))                # 记录非法
    ok = (len(bad) == 0)                               # 无非法
    ok_all &= ok
    print(f'  {_tag(ok)} RLE 格式合法（抽查 {len(sample)} 条，非法 {len(bad)}）')
    for b in bad[:3]:                                  # 打印前 3 个非法样例
        print(f'        {b[0]}: {b[1]}')

    # 6) 前景占比合理性
    means = []                                         # 前景占比列表
    for _, r in sample.iterrows():                     # 遍历抽样
        m = rle_decode(str(r['rle_mask']), H, W)       # 解码掩码
        means.append(m.mean())                         # 记录前景占比
    if means:                                          # 有数据
        mm = float(np.mean(means))                     # 平均前景占比
        ok = (0.05 < mm < 0.60)                        # 合理区间
        ok_all &= ok
        print(f'  {_tag(ok)} 预测前景占比均值 {mm:.3f}（合理区间 0.05~0.60）')
        if not ok:                                     # 不合理
            print(f'        ← 若接近 0 说明预测全背景；接近 1 说明全前景，模型或编码有问题')

    # 7) 全背景（'1 0'）数量检查
    n_empty = (df['rle_mask'].astype(str).str.strip() == '1 0').sum()  # 全背景数量
    ok = (n_empty < n * 0.05)                          # 全背景占比 < 5%
    ok_all &= ok
    print(f'  {_tag(ok)} 全背景图片 {n_empty} 张（占比 {n_empty/max(n,1):.2%}，应 < 5%）')
    if n_empty > n * 0.05:                             # 全背景过多
        print(f'        ← 大量全背景说明模型没学会，或对测试图读取失败')

    print(f'\n  >>> 自检 3 总体：{_tag(ok_all)}')
    return bool(ok_all)                                # 返回总体结果


def selftest_4_visual(df, img_dir, out_dir='./viz_check', n=4):
    """自检 4：肉眼可视化（可选，最有说服力）"""
    print('\n' + '=' * 70)
    print('自检 4：可视化抽查')
    print('=' * 70)
    try:                                               # 尝试导入 matplotlib
        import matplotlib
        matplotlib.use('Agg')                          # 无界面后端
        import matplotlib.pyplot as plt
    except Exception:                                  # 没有 matplotlib
        print('  ⚠ 没有 matplotlib，跳过可视化')
        return None

    os.makedirs(out_dir, exist_ok=True)                # 创建输出目录
    rows = df.head(n)                                  # 取前 n 行
    for i, (_, r) in enumerate(rows.iterrows()):       # 遍历每行
        name = str(r['img'])                           # img 列值
        p = os.path.join(img_dir, name if name.endswith('.jpg') else name + '.jpg')  # 图像路径
        if not os.path.exists(p):                      # 图像不存在
            alt = glob.glob(os.path.join(img_dir, '**', name + '*.jpg'), recursive=True)  # 递归查找
            if not alt:                                # 找不到
                continue
            p = alt[0]                                 # 用找到的路径
        img = Image.open(p).convert('RGB')             # 读取图像
        W, H = img.size                               # 原始宽高
        m = rle_decode(str(r['rle_mask']), H, W)       # 解码掩码

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))  # 3 列子图
        ax[0].imshow(img); ax[0].set_title('原图')     # 左：原图
        ax[1].imshow(m, cmap='gray'); ax[1].set_title(f'预测掩码 (前景{m.mean():.3f})')  # 中：掩码
        ax[2].imshow(img)                              # 右：叠加
        ax[2].imshow(np.ma.masked_where(m == 0, m), cmap='autumn', alpha=0.5)
        ax[2].set_title('叠加')
        for a in ax:                                   # 隐藏所有子图坐标轴
            a.axis('off')
        fp = os.path.join(out_dir, f'{i:02d}_{name}.png')  # 输出路径
        plt.tight_layout(); plt.savefig(fp, dpi=90); plt.close(fig)  # 保存
        print(f'  已保存: {fp}')
    print(f'  >>> 请打开 {out_dir}/ 下的图片肉眼确认掩码是否贴合车身')
    return True


# ===========================================================================
# 七、CLI（命令行接口）
# ===========================================================================
def get_args():
    """解析命令行参数"""
    ap = argparse.ArgumentParser(description='Carvana 批量推理 + 五级自检',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model', '-m', required=True, help='.pth 权重路径')  # 模型权重（必填）
    ap.add_argument('--test-dir', '-d', help='测试图目录')                 # 测试图目录
    ap.add_argument('--test-zip', '-z', help='测试图 zip（自动预解压到临时盘）')  # 测试图 zip
    ap.add_argument('--output', '-o', default='submission.csv', help='输出 CSV')  # 输出路径
    ap.add_argument('--batch-size', '-b', type=int, default=16, help='batch（多卡为总和）')  # 批次大小
    ap.add_argument('--scale', '-s', type=float, default=0.5, help='必须与训练一致')  # 缩放因子
    ap.add_argument('--classes', '-c', type=int, default=2)                # 类别数
    ap.add_argument('--no-amp', action='store_true', help='关闭混合精度（默认开启）')  # 关闭 AMP
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 4),
                    help='解码进程数')                                     # 数据加载进程数
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 张（试跑用）')  # 限制处理数量
    ap.add_argument('--selftest', action='store_true', help='只跑自检，不推理')  # 仅自检模式
    ap.add_argument('--viz', type=int, default=0, help='可视化前 N 张')     # 可视化数量
    ap.add_argument('--mask-dir', default='./data/masks', help='训练掩码目录（自检用）')  # 掩码目录
    ap.add_argument('--train-img-dir', default='./data/imgs', help='训练图目录（自检用）')  # 图像目录
    ap.add_argument('--official-csv', default=None, help='官方 train_masks.csv(.zip)')  # 官方 CSV
    ap.add_argument('--sample-sub', default=None, help='官方 sample_submission.csv（校验 img 集合）')  # 官方提交模板
    return ap.parse_args()                              # 解析并返回


def main():
    """主函数：自检 → 推理 → 格式检查"""
    args = get_args()                                   # 获取参数
    use_amp = not args.no_amp                           # 是否用 AMP（默认开启）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    print(f'[环境] device={device} | GPU={torch.cuda.device_count()} | amp={use_amp} | scale={args.scale}')

    # ---------- 自检 0：纯函数，永远先跑 ----------
    ok0 = selftest_0_rle_roundtrip()                    # RLE 编解码往返测试

    net, bilinear = load_model(args.model, args.classes, device)  # 加载模型

    # ---------- 自检 1 / 2 ----------
    ok1 = selftest_1_official_csv(args.official_csv, args.mask_dir)  # 与官方 CSV 比对
    ok2 = selftest_2_model_e2e(net, device, args.train_img_dir, args.mask_dir,
                               args.scale, n=20)        # 模型端到端 Dice

    if not ok0 or ok1 is False or ok2 is False:         # 自检未通过
        print('\n' + '!' * 70)
        print('! 自检未通过，禁止全量推理 —— 先修上面的 FAIL，否则 10 万张白跑')
        print('!' * 70)
        if args.selftest:                               # 仅自检模式
            return
        if input('\n仍要继续？(y/N): ').strip().lower() != 'y':  # 交互确认
            return

    if args.selftest:                                   # 仅自检模式
        print('\n自检模式结束（未执行推理）')
        return

    # ---------- 推理 ----------
    paths, _ = collect_paths(args.test_dir, args.test_zip)  # 收集测试图路径
    print(f'[数据] 找到 {len(paths)} 张测试图')
    assert len(paths) > 0, '没找到测试图，检查 --test-dir / --test-zip'
    if args.limit:                                      # 试跑模式
        paths = paths[:args.limit]
        print(f'[数据] 试跑模式：只处理前 {len(paths)} 张')

    # 官方 sample_submission 用于校验 img 集合
    expected_ids = None
    if args.sample_sub and os.path.exists(args.sample_sub):  # 有官方模板
        expected_ids = pd.read_csv(args.sample_sub)['img'].astype(str).tolist()  # 读取期望 ID
        print(f'[数据] 官方期望 {len(expected_ids)} 行')

    loader = build_loader(paths, args.scale, args.batch_size, args.workers)  # 构建加载器
    df = run_predict(net, loader, device, use_amp, args.output,
                     total_expected=len(expected_ids) if expected_ids else len(paths))  # 执行推理

    # ---------- 自检 3 / 4 ----------
    ok3 = selftest_3_format(df, expected_ids)           # 格式终检
    if args.viz:                                        # 可视化
        selftest_4_visual(df, args.test_dir or '/kaggle/tmp/test_images', n=args.viz)

    print('\n' + '=' * 70)
    if ok3:                                             # 格式检查通过
        print(f'✓ 全部通过，可以提交：{args.output}')
    else:                                               # 格式有问题
        print(f'✗ 格式有问题，修完再提交（见自检 3）')
    print('=' * 70)


if __name__ == '__main__':
    main()                                              # 程序入口
