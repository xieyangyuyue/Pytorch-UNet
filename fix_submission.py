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

import argparse  # 命令行参数解析
import os        # 操作系统接口
import sys       # 系统模块（本文件未直接使用）
import time      # 时间计时
from multiprocessing import Pool, cpu_count  # 多进程池 + CPU 核心数

import numpy as np   # 数值计算
import pandas as pd  # 数据表处理

H, W = 1280, 1918          # Carvana 原图尺寸 (高, 宽)
N = H * W                  # 总像素数


# ---------------------------------------------------------------------------
# 预计算置换表：F-flat 位置 i  →  C-flat 位置 j
#   i = col*H + row        (order='F' 展开)
#   j = row*W + col        (order='C' 展开)
# ---------------------------------------------------------------------------
def build_perm(h=H, w=W):
    """构建 F 序→C 序的位置置换表"""
    i = np.arange(h * w, dtype=np.int64)               # 0 到 H*W-1 的索引
    col = i // h                                       # F 序下的列号
    row = i % h                                        # F 序下的行号
    return row * w + col                               # C 序下的对应位置


PERM = None                # 全局置换表，子进程里惰性初始化


def _init_worker():
    """子进程初始化函数：构建置换表"""
    global PERM
    PERM = build_perm()


# ---------------------------------------------------------------------------
# RLE 编解码（向量化，比逐 run 循环快几十倍）
# ---------------------------------------------------------------------------
def rle_decode_flat(s, n=N):
    """RLE 字符串 → 1D flat 数组（长度 n）"""
    s = str(s).strip()                                 # 转字符串
    if s == '' or s == 'nan':                          # 空值
        return np.zeros(n, np.uint8)                   # 返回全零
    a = np.array([int(x) for x in s.split()], dtype=np.int64).reshape(-1, 2)  # 解析 (start, length)
    st, ln = a[:, 0] - 1, a[:, 1]                      # 转 0-based 起始 + 长度
    total = int(ln.sum())                              # 前景像素总数
    if total == 0:                                     # 全背景
        return np.zeros(n, np.uint8)
    # 一次性展开所有 run 的下标
    idx = np.repeat(st, ln) + (np.arange(total) - np.repeat(np.cumsum(ln) - ln, ln))
    flat = np.zeros(n, np.uint8)                       # 创建 1D 全零
    flat[idx] = 1                                      # 设置前景像素
    return flat


def rle_encode_flat(pixels, order='C'):
    """一维 uint8 数组 → RLE 字符串（按 order 视为 (H,W) 图像后展开）"""
    p = pixels.reshape(H, W).flatten(order=order)      # reshape 再按 order 展平
    p = np.concatenate([[0], p, [0]])                  # 两端补 0
    runs = np.where(p[1:] != p[:-1])[0] + 1            # 找跳变位置
    if runs.size == 0:                                 # 全背景
        return '1 0'
    runs[1::2] -= runs[::2]                            # 结束位置 → 长度
    return ' '.join(str(int(x)) for x in runs)         # 拼成字符串


def count_runs(s):
    """统计 RLE 字符串中的 run 数量（用于诊断）"""
    return len(str(s).split()) // 2                    # 数字数 / 2 = run 数


# ---------------------------------------------------------------------------
# 单条转换
# ---------------------------------------------------------------------------
def convert_one(s):
    """把一条 F 序提交串转成 C 序。"""
    flat = rle_decode_flat(s, N)                       # 解码为 1D（这是 M.flatten(order='F')）
    cflat = np.empty(N, np.uint8)                      # 创建目标数组
    cflat[PERM] = flat                                 # 置换成 M.flatten(order='C')
    return rle_encode_flat(cflat, 'C')                 # 用 C 序重新编码


def diagnose_one(s):
    """判断一条串更可能是 F 序还是 C 序。依据：正确解读下 run 数更少。"""
    flat = rle_decode_flat(s, N)                       # 解码为 1D
    if flat.sum() == 0:                                # 全背景无法判断
        return None
    # 假设 A：flat 是 F-flatten → 转成 C-flat 后编码
    cflat = np.empty(N, np.uint8)
    cflat[PERM] = flat
    n_from_F = count_runs(rle_encode_flat(cflat, 'C'))  # 假设 F 序的 run 数
    # 假设 B：flat 本来就是 C-flatten
    n_from_C = count_runs(rle_encode_flat(flat, 'C'))   # 假设 C 序的 run 数
    return n_from_F, n_from_C                          # run 数少的更可能是正确顺序


def _work(s):
    """多进程工作函数"""
    return convert_one(s)                              # 转换单条


# ---------------------------------------------------------------------------
def main():
    """主函数"""
    global H, W, N, PERM
    ap = argparse.ArgumentParser(description='F 序 submission.csv → C 序（无损，无需重跑推理）')
    ap.add_argument('csv', help='待修复的 submission.csv')               # 输入 CSV（位置参数）
    ap.add_argument('-o', '--output', default=None, help='输出路径')     # 输出路径
    ap.add_argument('--from-order', choices=['F', 'C'], default=None,
                    help='强制指定原编码顺序（默认自动检测）')            # 强制指定顺序
    ap.add_argument('--check-only', action='store_true', help='只诊断，不生成文件')  # 仅诊断
    ap.add_argument('--height', type=int, default=H)                     # 图像高度
    ap.add_argument('--width', type=int, default=W)                      # 图像宽度
    ap.add_argument('--workers', type=int, default=max(1, cpu_count() - 1))  # 进程数
    args = ap.parse_args()

    H, W = args.height, args.width                     # 更新全局尺寸
    N = H * W                                          # 总像素数
    PERM = build_perm(H, W)                            # 构建置换表

    print(f'读取 {args.csv} ...')
    t0 = time.time()                                   # 计时开始
    df = pd.read_csv(args.csv, dtype={'img': str, 'rle_mask': str})  # 读取 CSV
    print(f'  {len(df)} 行, 耗时 {time.time()-t0:.1f}s')

    # ---------------- 顺序检测 ----------------
    order = args.from_order                            # 是否强制指定
    if order is None:                                  # 自动检测
        print('\n自动检测编码顺序（抽样 200 条）...')
        sample = df['rle_mask'].dropna().sample(min(200, len(df)), random_state=0)  # 随机抽样
        nF = nC = 0                                    # F 序/C 序计数
        for s in sample:
            r = diagnose_one(s)                        # 诊断
            if r is None:                              # 全背景跳过
                continue
            a, b = r
            if a < b:                                  # F 序 run 更少
                nF += 1
            else:                                      # C 序 run 更少
                nC += 1
        print(f'  像 F 序(列优先): {nF} 条 | 像 C 序(行优先): {nC} 条')
        order = 'F' if nF > nC else 'C'               # 判定顺序
        print(f'  >>> 判定：原文件是 {order} 序')

    if order == 'C':                                   # 已经是 C 序
        print('\n✓ 原文件已经是 C 序（正确），无需转换。')
        print('  如果你的分数依然很低，问题不在编码顺序，请检查：')
        print('   1) 模型是否真的训好（验证集 Dice）')
        print('   2) --scale 是否与训练一致')
        print('   3) 掩码是否 resize 回了原图尺寸 1918x1280')
        return

    if args.check_only:                                # 仅诊断模式
        print('\n--check-only：不生成文件，结束。')
        return

    # ---------------- 转换 ----------------
    out = args.output or (os.path.splitext(args.csv)[0] + '_fixed.csv')  # 输出路径
    print(f'\n开始转换 {len(df)} 条 → {out} (workers={args.workers}) ...')
    t0 = time.time()                                   # 计时开始
    strs = df['rle_mask'].fillna('1 0').astype(str).tolist()  # 提取 RLE 字符串列表

    if args.workers > 1:                               # 多进程
        with Pool(args.workers, initializer=_init_worker) as pool:  # 创建进程池
            chunksize = 200                            # 每块大小
            results = []
            for i, r in enumerate(pool.imap(_work, strs, chunksize=chunksize)):  # 并行转换
                results.append(r)
                if (i + 1) % 5000 == 0:               # 每 5000 条打印进度
                    el = time.time() - t0
                    print(f'  [{i+1}/{len(strs)}] {el/60:.1f}min, '
                          f'预计剩余 {el/(i+1)*(len(strs)-i-1)/60:.1f}min')
    else:                                              # 单进程
        _init_worker()
        results = []
        for i, s in enumerate(strs):                   # 逐条转换
            results.append(_work(s))
            if (i + 1) % 5000 == 0:                   # 打印进度
                el = time.time() - t0
                print(f'  [{i+1}/{len(strs)}] {el/60:.1f}min, '
                      f'预计剩余 {el/(i+1)*(len(strs)-i-1)/60:.1f}min')

    fixed = pd.DataFrame({'img': df['img'], 'rle_mask': results})  # 构建结果 DataFrame
    fixed.to_csv(out, index=False)                     # 保存 CSV
    print(f'\n✓ 完成：{out}')
    print(f'  行数 {len(fixed)} | 大小 {os.path.getsize(out)/1024**2:.1f} MB '
          f'| 耗时 {(time.time()-t0)/60:.1f} min')

    # ---------------- 复核 ----------------
    print('\n复核（抽样 300 条，检查格式合法性）...')
    bad = 0                                            # 非法计数
    for s in fixed['rle_mask'].sample(min(300, len(fixed)), random_state=0):  # 抽样检查
        a = np.array([int(x) for x in str(s).split()], dtype=np.int64).reshape(-1, 2)
        if (a[:, 0] < 1).any() or (a[:, 1] < 0).any() or not (np.diff(a[:, 0]) > 0).all():  # 检查合法性
            bad += 1
    print(f'  非法 {bad} 条  {"✓ PASS" if bad == 0 else "✗ FAIL"}')
    print('\n前 3 行：')
    print(fixed.head(3).to_string())                   # 打印前 3 行


if __name__ == '__main__':
    main()                                              # 程序入口
