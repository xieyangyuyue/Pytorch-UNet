# 数据加载模块：定义数据集类 BasicDataset 和 CarvanaDataset
# 负责：读取图像/掩码文件、预处理（缩放、归一化、通道转换）、构建 PyTorch Dataset

# 导入日志模块，用于记录数据加载过程中的信息
import logging
# 导入 numpy，用于数组操作
import numpy as np
# 导入 PyTorch，用于将 numpy 数组转为张量
import torch
# 导入 PIL 图像库，用于读取和处理图像文件
from PIL import Image
# lru_cache: 最近最少使用缓存装饰器（本文件中导入但未使用）
from functools import lru_cache
# partial: 偏函数，固定部分参数生成新函数
from functools import partial
# repeat: 无限重复生成器（本文件中导入但未使用）
from itertools import repeat
# Pool: 多进程池，用于并行扫描掩码文件
from multiprocessing import Pool
# listdir: 列出目录内容
from os import listdir
# splitext: 分离文件名和扩展名
# isfile: 判断是否为文件
# join: 拼接路径
from os.path import splitext, isfile, join
# Path: 面向对象的路径操作
from pathlib import Path
# Dataset: PyTorch 数据集基类
from torch.utils.data import Dataset
# tqdm: 进度条
from tqdm import tqdm


def load_image(filename):
    # 根据文件扩展名加载图像，返回 PIL.Image 对象
    # filename: 图像文件路径
    ext = splitext(filename)[1]  # 提取文件扩展名
    if ext == '.npy':  # 如果是 numpy 数组文件
        return Image.fromarray(np.load(filename))  # 加载 numpy 数组并转为 PIL 图像
    elif ext in ['.pt', '.pth']:  # 如果是 PyTorch 张量文件 (.pt/.pth)
        return Image.fromarray(torch.load(filename).numpy())  # 加载张量并转为 PIL 图像
    else:  # 其他格式（.jpg, .png, .gif 等）
        return Image.open(filename)  # 直接用 PIL 打开


def unique_mask_values(idx, mask_dir, mask_suffix):
    # 扫描单个掩码文件，返回其中所有唯一的像素值
    # idx:        图像 ID（不含扩展名）
    # mask_dir:   掩码目录
    # mask_suffix: 掩码文件后缀（如 Carvana 的 '_mask'）
    # 用 glob 匹配掩码文件（如 "123_mask.gif"），取第一个匹配结果
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    # 加载掩码并转为 numpy 数组
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:  # 二维掩码（灰度图，每个像素值代表类别）
        return np.unique(mask)  # 返回所有唯一像素值
    elif mask.ndim == 3:  # 三维掩码（RGB 图，每个颜色代表类别）
        mask = mask.reshape(-1, mask.shape[-1])  # 展平为 (像素数, 通道数)
        return np.unique(mask, axis=0)  # 返回所有唯一颜色
    else:  # 其他维度不支持
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    # 通用数据集类：从目录加载图像和掩码，支持缩放和预处理
    # 继承 PyTorch 的 Dataset，实现 __len__ 和 __getitem__

    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        # images_dir:  图像目录路径
        # mask_dir:    掩码目录路径
        # scale:       缩放因子 (0, 1]，0.5 表示缩小到一半
        # mask_suffix: 掩码文件名后缀（如 Carvana 用 '_mask'）
        self.images_dir = Path(images_dir)  # 转为 Path 对象
        self.mask_dir = Path(mask_dir)      # 转为 Path 对象
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'  # 缩放因子必须在 (0, 1]
        self.scale = scale  # 保存缩放因子
        self.mask_suffix = mask_suffix  # 保存掩码后缀

        # 扫描图像目录，提取所有非隐藏文件的 ID（不含扩展名）
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:  # 如果没有找到任何图像文件
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')  # 记录数据集大小
        logging.info('Scanning mask files to determine unique values')  # 记录开始扫描掩码
        # 使用多进程并行扫描所有掩码的唯一像素值
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)  # 进度条总数
            ))

        # 将所有掩码的唯一值合并、排序，得到全局类别列表
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')  # 记录类别值

    def __len__(self):
        # 返回数据集大小（图像数量）
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        # 预处理：缩放 + 通道转换 + 归一化
        # mask_values: 全局类别值列表
        # pil_img:     PIL 图像对象
        # scale:       缩放因子
        # is_mask:     是否为掩码（掩码用最近邻插值，图像用双三次插值）
        w, h = pil_img.size  # 获取原始宽高
        newW, newH = int(scale * w), int(scale * h)  # 计算缩放后的宽高
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'  # 确保缩放后仍有像素
        # 缩放：掩码用 NEAREST（最近邻，保持类别值不变），图像用 BICUBIC（双三次，平滑）
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)  # 转为 numpy 数组

        if is_mask:  # 如果是掩码
            # 创建 int64 类型的掩码数组，初始全 0
            mask = np.zeros((newH, newW), dtype=np.int64)
            # 将每个原始像素值映射为类别索引（0, 1, 2, ...）
            for i, v in enumerate(mask_values):
                if img.ndim == 2:  # 灰度掩码
                    mask[img == v] = i  # 像素值等于 v 的位置设为类别索引 i
                else:  # RGB 掩码
                    mask[(img == v).all(-1)] = i  # 颜色匹配的位置设为类别索引 i

            return mask  # 返回处理后的掩码

        else:  # 如果是图像
            if img.ndim == 2:  # 灰度图
                img = img[np.newaxis, ...]  # 添加通道维度：(H, W) → (1, H, W)
            else:  # RGB 图
                img = img.transpose((2, 0, 1))  # HWC → CHW：(H, W, 3) → (3, H, W)

            if (img > 1).any():  # 如果像素值大于 1（即 0-255 范围）
                img = img / 255.0  # 归一化到 [0, 1]

            return img  # 返回处理后的图像

    def __getitem__(self, idx):
        # 按索引获取一个样本（图像 + 掩码）
        # idx: 样本索引
        name = self.ids[idx]  # 获取样本 ID
        # 用 glob 匹配掩码文件（支持任意扩展名）
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        # 用 glob 匹配图像文件（支持任意扩展名）
        img_file = list(self.images_dir.glob(name + '.*'))

        # 断言：图像文件必须唯一存在
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # 断言：掩码文件必须唯一存在
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])  # 加载掩码
        img = load_image(img_file[0])    # 加载图像

        # 断言：图像和掩码尺寸必须一致
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # 预处理图像（缩放 + CHW + 归一化）
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        # 预处理掩码（缩放 + 类别映射）
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        # 返回字典：图像张量 (float32, contiguous) + 掩码张量 (long, contiguous)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),  # 图像转为 float32 张量
            'mask': torch.as_tensor(mask.copy()).long().contiguous()    # 掩码转为 long 张量（类别索引）
        }


class CarvanaDataset(BasicDataset):
    # Carvana 数据集：继承 BasicDataset，固定掩码后缀为 '_mask'
    # Carvana 比赛数据：图像为 xxx.jpg，掩码为 xxx_mask.gif
    def __init__(self, images_dir, mask_dir, scale=1):
        # 调用父类初始化，mask_suffix='_mask'
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
