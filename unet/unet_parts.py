""" U-Net 模型的组成部分 """
# 本文件定义 U-Net 网络的基本构件：DoubleConv、Down、Up、OutConv
# U-Net 论文: https://arxiv.org/abs/1505.04597

# 导入 PyTorch 核心库
import torch  # PyTorch 主库，提供张量运算
import torch.nn as nn  # 神经网络模块，包含各种层（Conv2d、BatchNorm2d、ReLU 等）
import torch.nn.functional as F  # 函数式接口，用于无参数操作（如 padding、interpolate）


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    # 双卷积块：两次 "卷积 → 批归一化 → ReLU 激活" 的组合
    # 这是 U-Net 中最基础的重复单元，出现在编码器和解码器的每一层

    def __init__(self, in_channels, out_channels, mid_channels=None):
        # in_channels:  输入通道数
        # out_channels: 输出通道数
        # mid_channels: 中间层通道数（默认等于 out_channels）
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        if not mid_channels:  # 如果未指定中间通道数
            mid_channels = out_channels  # 则中间通道数等于输出通道数
        # 定义双卷积的顺序容器：Conv → BN → ReLU → Conv → BN → ReLU
        self.double_conv = nn.Sequential(
            # 第一个卷积：3×3 卷积核，padding=1 保持尺寸不变，bias=False 因为后面有 BN
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # 批归一化：加速收敛、稳定训练
            nn.BatchNorm2d(mid_channels),
            # ReLU 激活函数：inplace=True 节省内存（直接在输入张量上操作）
            nn.ReLU(inplace=True),
            # 第二个卷积：同样 3×3，padding=1，无偏置
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 第二个批归一化
            nn.BatchNorm2d(out_channels),
            # 第二个 ReLU 激活
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 前向传播：将输入 x 依次通过双卷积块的所有层
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    # 下采样块：先 2×2 最大池化降分辨率，再做双卷积提取特征
    # 这是 U-Net 编码器（左侧收缩路径）的每一层

    def __init__(self, in_channels, out_channels):
        # in_channels:  输入通道数
        # out_channels: 输出通道数（通常翻倍）
        super().__init__()  # 父类初始化
        # 定义下采样的顺序容器：MaxPool → DoubleConv
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2×2 最大池化，空间尺寸减半（H/2, W/2）
            DoubleConv(in_channels, out_channels)  # 双卷积提取特征
        )

    def forward(self, x):
        # 前向传播：先池化降采样，再双卷积
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    # 上采样块：先上采样提高分辨率，再与跳跃连接的特征拼接，最后双卷积
    # 这是 U-Net 解码器（右侧扩展路径）的每一层

    def __init__(self, in_channels, out_channels, bilinear=True):
        # in_channels:  输入通道数（包含拼接后的通道）
        # out_channels: 输出通道数
        # bilinear:     是否使用双线性插值上采样（True）或转置卷积（False）
        super().__init__()  # 父类初始化

        # 如果使用双线性插值上采样
        if bilinear:
            # 双线性插值：将特征图放大 2 倍，align_corners=True 对齐角点
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 双卷积：输入通道减半（因为双线性不改变通道数，拼接后通道为 in_channels）
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 转置卷积：学习上采样，通道数减半，kernel_size=2 stride=2 实现 2 倍上采样
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # 双卷积：输入通道为 in_channels（上采样后 + 跳跃连接拼接）
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: 来自下一层的上采样特征（需要放大）
        # x2: 来自编码器的跳跃连接特征（同分辨率）
        x1 = self.up(x1)  # 上采样 x1，空间尺寸翻倍
        # 以下处理尺寸不匹配的 padding 问题（当输入尺寸为奇数时，上采样后可能差 1 像素）
        # input is CHW (Channel, Height, Width)
        diffY = x2.size()[2] - x1.size()[2]  # 高度差：x2 的高 - x1 的高
        diffX = x2.size()[3] - x1.size()[3]  # 宽度差：x2 的宽 - x1 的宽

        # 对 x1 进行对称 padding，使其尺寸与 x2 完全一致
        # padding 顺序: [左, 右, 上, 下]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # 在通道维度上拼接：x2（跳跃连接）在前，x1（上采样）在后
        x = torch.cat([x2, x1], dim=1)  # dim=1 表示通道维度拼接
        # 拼接后通过双卷积融合特征
        return self.conv(x)


class OutConv(nn.Module):
    # 输出卷积层：1×1 卷积，将通道数映射到类别数
    # 这是 U-Net 最后一层，把特征图转换为每个像素的类别概率（logits）
    def __init__(self, in_channels, out_channels):
        # in_channels:  输入通道数（来自最后一层 Up 的输出）
        # out_channels: 输出通道数 = 类别数（n_classes）
        super(OutConv, self).__init__()  # 父类初始化
        # 1×1 卷积：不改变空间尺寸，仅做通道映射
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 前向传播：1×1 卷积输出 logits
        return self.conv(x)
