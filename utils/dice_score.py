# 导入 PyTorch 张量类型，用于类型注解
import torch
from torch import Tensor  # Tensor 是 PyTorch 的核心数据结构（多维数组）


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Dice 系数计算：衡量预测掩码与真实掩码的相似度
    # 公式: Dice = 2 * |A ∩ B| / (|A| + |B|)
    # input:  预测掩码张量
    # target: 真实掩码张量
    # reduce_batch_first: 是否在 batch 维度上先归约
    # epsilon: 防止除零的小常数
    # 断言：预测和真实掩码形状必须一致
    assert input.size() == target.size()
    # 断言：如果要在 batch 维度归约，输入至少是 3 维
    assert input.dim() == 3 or not reduce_batch_first

    # 确定归约维度：
    # 如果输入是 2 维（单张图）或不在 batch 维度归约，只对最后两个维度（H, W）求和
    # 否则对最后三个维度（B, H, W）求和
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # 交集：2 * (预测 × 真实) 的逐元素乘积之和
    inter = 2 * (input * target).sum(dim=sum_dim)
    # 并集近似：预测之和 + 真实之和
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    # 当两者都为 0 时（空掩码），将分母设为 inter（即 0），避免 0/0
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # Dice = (2 * 交集 + epsilon) / (并集 + epsilon)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    # 返回所有样本/类别的平均 Dice
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # 多分类 Dice 系数：将 (B, C, H, W) 展平为 (B*C, H, W) 后调用二分类版本
    # input:  预测掩码，形状 (B, C, H, W)，C 为类别数
    # target: 真实掩码，形状 (B, C, H, W)
    # 将 batch 和 channel 维度展平：把每个 batch 的每个类别视为独立样本
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice 损失：1 - Dice 系数，值域 [0, 1]，越小越好
    # input:  预测掩码
    # target: 真实掩码
    # multiclass: 是否为多分类任务
    # 根据是否多分类选择对应的 Dice 系数函数
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    # 损失 = 1 - Dice，reduce_batch_first=True 在 batch 维度归约
    return 1 - fn(input, target, reduce_batch_first=True)
