""" Full assembly of the parts to form the complete network """
# 将 unet_parts.py 中定义的组件组装成完整的 U-Net 网络

# 导入所有组件类：DoubleConv, Down, Up, OutConv
from .unet_parts import *


class UNet(nn.Module):
    # U-Net 网络的完整实现
    # 结构：编码器（4 层下采样）→ 瓶颈层 → 解码器（4 层上采样）→ 输出层
    # 编码器每层通道数翻倍，解码器每层通道数减半，通过跳跃连接保留细节

    def __init__(self, n_channels, n_classes, bilinear=False):
        # n_channels: 输入图像的通道数（RGB=3，灰度=1）
        # n_classes:  输出类别数（二分类=2，多分类=类别数）
        # bilinear:   是否使用双线性插值上采样（True）或转置卷积（False）
        super(UNet, self).__init__()  # 父类 nn.Module 初始化
        self.n_channels = n_channels  # 记录输入通道数，供外部访问
        self.n_classes = n_classes    # 记录输出类别数，供外部访问
        self.bilinear = bilinear      # 记录上采样方式，供外部访问

        # ===== 编码器（收缩路径）=====
        self.inc = (DoubleConv(n_channels, 64))      # 初始双卷积：3→64 通道
        self.down1 = (Down(64, 128))                  # 下采样 1：64→128 通道
        self.down2 = (Down(128, 256))                 # 下采样 2：128→256 通道
        self.down3 = (Down(256, 512))                 # 下采样 3：256→512 通道
        # factor: 双线性时通道减半因子=2，转置卷积时=1
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))     # 瓶颈层下采样：512→512(双线性) 或 512→1024(转置卷积)

        # ===== 解码器（扩展路径）=====
        self.up1 = (Up(1024, 512 // factor, bilinear))  # 上采样 1：1024→256(双线性) 或 1024→512(转置卷积)
        self.up2 = (Up(512, 256 // factor, bilinear))   # 上采样 2：512→128(双线性) 或 512→256(转置卷积)
        self.up3 = (Up(256, 128 // factor, bilinear))   # 上采样 3：256→64(双线性) 或 256→128(转置卷积)
        self.up4 = (Up(128, 64, bilinear))               # 上采样 4：128→64（两种方式都一样）
        self.outc = (OutConv(64, n_classes))             # 输出层：64→n_classes 的 1×1 卷积

    def forward(self, x):
        # 前向传播：U-Net 的完整数据流
        # x: 输入张量，形状 (B, n_channels, H, W)
        x1 = self.inc(x)       # x1: 初始特征，尺寸 H×W，64 通道（跳跃连接保留）
        x2 = self.down1(x1)    # x2: 下采样 1，尺寸 H/2×W/2，128 通道（跳跃连接保留）
        x3 = self.down2(x2)    # x3: 下采样 2，尺寸 H/4×W/4，256 通道（跳跃连接保留）
        x4 = self.down3(x3)    # x4: 下采样 3，尺寸 H/8×W/8，512 通道（跳跃连接保留）
        x5 = self.down4(x4)    # x5: 瓶颈层，尺寸 H/16×W/16，512/1024 通道
        # 解码器：每层上采样后与对应编码器特征拼接（跳跃连接）
        x = self.up1(x5, x4)   # 上采样 1：x5 上采样后与 x4 拼接
        x = self.up2(x, x3)    # 上采样 2：与 x3 拼接
        x = self.up3(x, x2)    # 上采样 3：与 x2 拼接
        x = self.up4(x, x1)    # 上采样 4：与 x1 拼接
        logits = self.outc(x)  # 输出层：1×1 卷积得到每个像素的类别 logits
        return logits          # 返回 logits，形状 (B, n_classes, H, W)

    def use_checkpointing(self):
        # 启用梯度检查点（gradient checkpointing）
        # 用时间换空间：不保存中间激活值，反向传播时重新计算
        # 适用于 GPU 显存不足的情况
        self.inc = torch.utils.checkpoint(self.inc)    # 初始双卷积启用检查点
        self.down1 = torch.utils.checkpoint(self.down1)  # 下采样 1 启用检查点
        self.down2 = torch.utils.checkpoint(self.down2)  # 下采样 2 启用检查点
        self.down3 = torch.utils.checkpoint(self.down3)  # 下采样 3 启用检查点
        self.down4 = torch.utils.checkpoint(self.down4)  # 瓶颈层启用检查点
        self.up1 = torch.utils.checkpoint(self.up1)    # 上采样 1 启用检查点
        self.up2 = torch.utils.checkpoint(self.up2)    # 上采样 2 启用检查点
        self.up3 = torch.utils.checkpoint(self.up3)    # 上采样 3 启用检查点
        self.up4 = torch.utils.checkpoint(self.up4)    # 上采样 4 启用检查点
        self.outc = torch.utils.checkpoint(self.outc)  # 输出层启用检查点
