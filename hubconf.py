# torch.hub 配置文件：允许通过 torch.hub.load() 加载预训练模型
# 用法: net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)

# 导入 PyTorch
import torch
# 导入项目内的 UNet 模型类，别名 _UNet 避免与本文件的函数名冲突
from unet import UNet as _UNet

def unet_carvana(pretrained=False, scale=0.5):
    """
    UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-image-masking-challenge/data ).
    Set the scale to 0.5 (50%) when predicting.
    """
    # 创建 UNet 模型：3 通道输入（RGB），2 类输出（前景/背景），转置卷积上采样
    net = _UNet(n_channels=3, n_classes=2, bilinear=False)
    if pretrained:  # 如果需要加载预训练权重
        if scale == 0.5:  # 0.5 倍缩放的预训练模型
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth'
        elif scale == 1.0:  # 1.0 倍缩放（原始分辨率）的预训练模型
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth'
        else:  # 其他缩放比例没有预训练模型
            raise RuntimeError('Only 0.5 and 1.0 scales are available')
        # 从 URL 下载权重文件
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        if 'mask_values' in state_dict:  # 如果权重中包含 mask_values（数据集信息，非模型参数）
            state_dict.pop('mask_values')  # 移除它，否则 load_state_dict 会报错
        net.load_state_dict(state_dict)  # 加载权重到模型

    return net  # 返回模型
