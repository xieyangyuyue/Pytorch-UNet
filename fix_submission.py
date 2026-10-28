#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_submission.py —— 把已生成的 F 序（列优先）submission.csv 无损转成正确的 C 序

适用场景：
  你用 batch_predict.py（rle_encode 里写 order='F'）已经跑完了 10 万张推理，
  花了 2.5 小时，结果提交只有 0.36 分。本脚本直接把 CSV 修好，
  【不需要重新跑推理】，几分钟就能出正确的 submission_fixed.csv。

原理：
  提交串 S = RLE( M.flatten(order='F') )
  官方解码  = 把 S 还原成一维 flat 数组（长度 H*W）
  所以      M = flat.reshape(H, W, order='F')
  然后      S_correct = RLE( M.flatten(order='C') )
  全程只是下标置换，零信息损失。

用法：
  python fix_submission.py submission.csv                    # 自动检测顺序
  python fix_submission.py submission.csv -o fixed.csv
  python fix_submission.py submission.csv --from-order F     # 强制指定原顺序
  python fix_submission.py submission.csv --check-only       # 只诊断不输出
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

H, W = 1280, 1918          # Carvana 原图尺寸 (高, 宽)
N = H * W


# ---------------------------------------------------------------------------
# 预计算置换表：F-flat 位置 i  →  C-flat 位置 j
#   i = col*H + row        (order='F' 展开)
#   j = row*W + col        (order='C' 展开)
# ---------------------------------------------------------------------------
def build_perm(h=H, w=W):
    i = np.arange(h * w, dtype=np.int64)
    col = i // h
    row = i % h
    return row * w + col


PERM = None                # 子进程里惰性初始化


def _init_worker():
    global PERM
    PERM = build_perm()


# ---------------------------------------------------------------------------
# RLE 编解码（向量化，比逐 run 循环快几十倍）
# ---------------------------------------------------------------------------
def rle_decode_flat(s, n=N):
    s = str(s).strip()
    if s == '' or s == 'nan':
        return np.zeros(n, np.uint8)
    a = np.array([int(x) for x in s.split()], dtype=np.int64).reshape(-1, 2)
    st, ln = a[:, 0] - 1, a[:, 1]
    total = int(ln.sum())
    if total == 0:
        return np.zeros(n, np.uint8)
    # 一次性展开所有 run 的下标：[st, st+1, ..., st+ln-1]
    idx = np.repeat(st, ln) + (np.arange(total) - np.repeat(np.cumsum(ln) - ln, ln))
    flat = np.zeros(n, np.uint8)
    flat[idx] = 1
    return flat


def rle_encode_flat(pixels, order='C'):
    """一维 uint8 数组 → RLE 字符串（按 order 视为 (H,W) 图像后展开）"""
    p = pixels.reshape(H, W).flatten(order=order)
    p = np.concatenate([[0], p, [0]])
    runs = np.where(p[1:] != p[:-1])[0] + 1
    if runs.size == 0:
        return '1 0'
    runs[1::2] -= runs[::2]
    return ' '.join(str(int(x)) for x in runs)


def count_runs(s):
    return len(str(s).split()) // 2


# ---------------------------------------------------------------------------
# 单条转换
# ---------------------------------------------------------------------------
def convert_one(s):
    """把一条 F 序提交串转成 C 序。"""
    flat = rle_decode_flat(s, N)          # 这就是 M.flatten(order='F')
    cflat = np.empty(N, np.uint8)
    cflat[PERM] = flat                    # 置换成 M.flatten(order='C')
    return rle_encode_flat(cflat, 'C')


def diagnose_one(s):
    """判断一条串更可能是 F 序还是 C 序。

    依据：真实掩码是连通块，正确解读下重编码的 run 数更少；
    错误解读会得到散乱的斜向碎片，run 数暴涨。
    """
    flat = rle_decode_flat(s, N)
    if flat.sum() == 0:
        return None
    # 假设 A：flat 是 F-flatten → 转成 C-flat 后编码
    cflat = np.empty(N, np.uint8)
    cflat[PERM] = flat
    n_from_F = count_runs(rle_encode_flat(cflat, 'C'))
    # 假设 B：flat 本来就是 C-flatten
    n_from_C = count_runs(rle_encode_flat(flat, 'C'))
    return n_from_F, n_from_C


def _work(s):
    return convert_one(s)


# ---------------------------------------------------------------------------
def main():
    global H, W, N, PERM
    ap = argparse.ArgumentParser(description='F 序 submission.csv → C 序（无损，无需重跑推理）')
    ap.add_argument('csv', help='待修复的 submission.csv')
    ap.add_argument('-o', '--output', default=None, help='输出路径（默认 submission_fixed.csv）')
    ap.add_argument('--from-order', choices=['F', 'C'], default=None,
                    help='强制指定原编码顺序（默认自动检测）')
    ap.add_argument('--check-only', action='store_true', help='只诊断，不生成文件')
    ap.add_argument('--height', type=int, default=H)
    ap.add_argument('--width', type=int, default=W)
    ap.add_argument('--workers', type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    H, W = args.height, args.width
    N = H * W
    PERM = build_perm(H, W)

    print(f'读取 {args.csv} ...')
    t0 = time.time()
    df = pd.read_csv(args.csv, dtype={'img': str, 'rle_mask': str})
    print(f'  {len(df)} 行, 耗时 {time.time()-t0:.1f}s')

    # ---------------- 顺序检测 ----------------
    order = args.from_order
    if order is None:
        print('\n自动检测编码顺序（抽样 200 条）...')
        sample = df['rle_mask'].dropna().sample(min(200, len(df)), random_state=0)
        nF = nC = 0
        for s in sample:
            r = diagnose_one(s)
            if r is None:
                continue
            a, b = r
            if a < b:
                nF += 1
            else:
                nC += 1
        print(f'  像 F 序(列优先): {nF} 条 | 像 C 序(行优先): {nC} 条')
        order = 'F' if nF > nC else 'C'
        print(f'  >>> 判定：原文件是 {order} 序')

    if order == 'C':
        print('\n✓ 原文件已经是 C 序（正确），无需转换。')
        print('  如果你的分数依然很低，问题不在编码顺序，请检查：')
        print('   1) 模型是否真的训好（验证集 Dice）')
        print('   2) --scale 是否与训练一致')
        print('   3) 掩码是否 resize 回了原图尺寸 1918x1280')
        return

    if args.check_only:
        print('\n--check-only：不生成文件，结束。')
        return

    # ---------------- 转换 ----------------
    out = args.output or (os.path.splitext(args.csv)[0] + '_fixed.csv')
    print(f'\n开始转换 {len(df)} 条 → {out} (workers={args.workers}) ...')
    t0 = time.time()
    strs = df['rle_mask'].fillna('1 0').astype(str).tolist()

    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker) as pool:
            chunksize = 200
            results = []
            for i, r in enumerate(pool.imap(_work, strs, chunksize=chunksize)):
                results.append(r)
                if (i + 1) % 5000 == 0:
                    el = time.time() - t0
                    print(f'  [{i+1}/{len(strs)}] {el/60:.1f}min, '
                          f'预计剩余 {el/(i+1)*(len(strs)-i-1)/60:.1f}min')
    else:
        _init_worker()
        results = []
        for i, s in enumerate(strs):
            results.append(_work(s))
            if (i + 1) % 5000 == 0:
                el = time.time() - t0
                print(f'  [{i+1}/{len(strs)}] {el/60:.1f}min, '
                      f'预计剩余 {el/(i+1)*(len(strs)-i-1)/60:.1f}min')

    fixed = pd.DataFrame({'img': df['img'], 'rle_mask': results})
    fixed.to_csv(out, index=False)
    print(f'\n✓ 完成：{out}')
    print(f'  行数 {len(fixed)} | 大小 {os.path.getsize(out)/1024**2:.1f} MB '
          f'| 耗时 {(time.time()-t0)/60:.1f} min')

    # ---------------- 复核 ----------------
    print('\n复核（抽样 300 条，检查格式合法性）...')
    bad = 0
    for s in fixed['rle_mask'].sample(min(300, len(fixed)), random_state=0):
        a = np.array([int(x) for x in str(s).split()], dtype=np.int64).reshape(-1, 2)
        if (a[:, 0] < 1).any() or (a[:, 1] < 0).any() or not (np.diff(a[:, 0]) > 0).all():
            bad += 1
    print(f'  非法 {bad} 条  {"✓ PASS" if bad == 0 else "✗ FAIL"}')
    print('\n前 3 行：')
    print(fixed.head(3).to_string())


if __name__ == '__main__':
    main()
