# 评估模块：在验证集上计算 Dice 系数，用于训练过程中监控模型性能

# 导入 PyTorch 核心库
import torch
# 导入函数式接口，用于 sigmoid、interpolate、one_hot 等无参数操作
import torch.nn.functional as F
# 导入进度条
from tqdm import tqdm

# 从 dice_score 模块导入多分类和二分类的 Dice 系数计算函数
from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()  # 装饰器：禁用梯度计算，节省显存和加速推理
def evaluate(net, dataloader, device, amp):
    # 在验证集上评估模型的 Dice 系数
    # net:        U-Net 模型
    # dataloader: 验证集数据加载器
    # device:     计算设备 (cuda/cpu)
    # amp:        是否使用混合精度
    net.eval()  # 切换到评估模式（关闭 Dropout、BN 使用全局统计量）
    num_val_batches = len(dataloader)  # 验证集 batch 数量
    dice_score = 0  # 累计 Dice 分数

    # iterate over the validation set
    # 在验证集上迭代，使用混合精度上下文
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        # 遍历验证集的每个 batch
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']  # 取出图像和真实掩码

            # move images and labels to correct device and type
            # 将数据移到 GPU，图像转 float32 并使用 channels_last 内存格式加速
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # 掩码转 long（类别索引）
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)  # 前向传播得到预测 logits

            if net.n_classes == 1:  # 二分类（单通道输出）
                # 断言：真实掩码值应在 [0, 1] 范围内
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                # sigmoid → 阈值 0.5 → 二值掩码
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                # 计算二分类 Dice 系数
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:  # 多分类
                # 断言：真实掩码值应在 [0, n_classes) 范围内
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                # 将真实掩码转为 one-hot 编码：(B, H, W) → (B, H, W, C) → (B, C, H, W)
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                # 将预测 logits 通过 argmax 转为类别，再转 one-hot：(B, C, H, W)
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                # 计算多分类 Dice，忽略背景类（第 0 类）
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()  # 切换回训练模式
    # 返回平均 Dice 系数（总分数 / batch 数）
    return dice_score / max(num_val_batches, 1)
