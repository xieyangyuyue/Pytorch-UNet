# 训练脚本：训练 U-Net 模型进行图像分割
# 用法: python train.py --epochs 5 --batch-size 1 --learning-rate 1e-5 --scale 0.5 --amp

# 导入命令行参数解析模块
import argparse
# 导入日志模块，用于记录训练信息
import logging
# 导入操作系统接口，用于获取 CPU 核心数
import os
# 导入随机数模块（本文件中导入但未直接使用）
import random
# 导入系统模块（本文件中导入但未直接使用）
import sys
# 导入 PyTorch 核心库
import torch
# 导入神经网络模块（损失函数等）
import torch.nn as nn
# 导入函数式接口（softmax、one_hot 等）
import torch.nn.functional as F
# 导入 torchvision 变换（本文件中导入但未直接使用）
import torchvision.transforms as transforms
# 导入 torchvision 函数式变换（本文件中导入但未直接使用）
import torchvision.transforms.functional as TF
# 导入路径处理模块
from pathlib import Path
# 导入优化器模块
from torch import optim
# 导入数据加载器和数据集划分工具
from torch.utils.data import DataLoader, random_split
# 导入进度条
from tqdm import tqdm

# 导入 Weights & Biases 实验跟踪工具
import wandb
# 导入评估函数
from evaluate import evaluate
# 导入 U-Net 模型
from unet import UNet
# 导入数据集类
from utils.data_loading import BasicDataset, CarvanaDataset
# 导入 Dice 损失函数
from utils.dice_score import dice_loss

# 默认路径定义
dir_img = Path('./data/imgs/')          # 训练图像目录
dir_mask = Path('./data/masks/')        # 训练掩码目录
dir_checkpoint = Path('./checkpoints/')  # 模型检查点保存目录


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 训练 U-Net 模型的主函数
    # model:          U-Net 模型实例
    # device:         计算设备 (cuda/cpu)
    # epochs:         训练轮数
    # batch_size:     批次大小
    # learning_rate:  学习率
    # val_percent:    验证集占比 (0-1)
    # save_checkpoint: 是否保存检查点
    # img_scale:      图像缩放因子
    # amp:            是否使用混合精度
    # weight_decay:   权重衰减（L2 正则化系数）
    # momentum:       RMSprop 动量
    # gradient_clipping: 梯度裁剪阈值

    # 1. Create dataset
    # 创建数据集：先尝试 CarvanaDataset，失败则回退到 BasicDataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)  # 尝试 Carvana 数据集（掩码后缀 _mask）
    except (AssertionError, RuntimeError, IndexError):  # 如果 Carvana 格式不匹配
        dataset = BasicDataset(dir_img, dir_mask, img_scale)  # 回退到通用数据集

    # 2. Split into train / validation partitions
    # 按比例划分训练集和验证集
    n_val = int(len(dataset) * val_percent)  # 验证集样本数
    n_train = len(dataset) - n_val           # 训练集样本数
    # 随机划分，固定随机种子 0 保证可复现
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    # 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)  # 公共参数：批次大小、多进程加载、锁页内存
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)  # 训练集：打乱顺序
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)  # 验证集：不打乱，丢弃最后不完整的 batch

    # (Initialize logging)
    # 初始化 W&B 实验跟踪
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')  # 创建匿名实验
    # 记录超参数到 W&B
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    # 记录训练配置信息
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # 配置优化器：RMSprop，带权重衰减和动量
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # 学习率调度器：验证 Dice 不提升时降低学习率，耐心值 5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # 混合精度的梯度缩放器（防止 FP16 梯度下溢）
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp) if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler(enabled=amp)
    # 损失函数：多分类用 CrossEntropyLoss，二分类用 BCEWithLogitsLoss
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0  # 全局步数计数器

    # 5. Begin training
    # 开始训练循环
    for epoch in range(1, epochs + 1):  # 遍历每个 epoch
        model.train()  # 切换到训练模式
        epoch_loss = 0  # 当前 epoch 的累计损失
        # 创建进度条
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:  # 遍历训练集每个 batch
                images, true_masks = batch['image'], batch['mask']  # 取出图像和真实掩码

                # 断言：图像通道数必须与模型定义的输入通道数一致
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # 将数据移到 GPU：图像用 float32 + channels_last 内存格式
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)  # 掩码用 long（类别索引）

                # 混合精度上下文：自动选择 FP16/FP32
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)  # 前向传播得到预测 logits
                    if model.n_classes == 1:  # 二分类
                        # BCE 损失 + Dice 损失
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:  # 多分类
                        # CrossEntropy 损失 + Dice 损失
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),  # softmax 转概率
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),  # one-hot 编码
                            multiclass=True
                        )

                # 反向传播
                optimizer.zero_grad(set_to_none=True)  # 清零梯度（set_to_none 比 zero 更快）
                grad_scaler.scale(loss).backward()  # 缩放损失并反向传播
                grad_scaler.unscale_(optimizer)  # 反缩放梯度，以便梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)  # 梯度裁剪
                grad_scaler.step(optimizer)  # 更新参数
                grad_scaler.update()  # 更新缩放因子

                pbar.update(images.shape[0])  # 更新进度条
                global_step += 1  # 全局步数 +1
                epoch_loss += loss.item()  # 累计损失
                # 记录训练损失到 W&B
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})  # 在进度条显示当前 batch 损失

                # Evaluation round
                # 定期评估：每 epoch 评估 5 次
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # 记录权重和梯度的直方图到 W&B
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')  # W&B 标签中 / 有特殊含义，替换为 .
                            if not (torch.isinf(value) | torch.isnan(value)).any():  # 跳过 inf/nan 权重
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():  # 跳过 inf/nan 梯度
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # 在验证集上评估 Dice 系数
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)  # 根据验证分数调整学习率

                        logging.info('Validation Dice score: {}'.format(val_score))  # 记录验证分数
                        try:
                            # 记录验证结果到 W&B（图像、掩码、学习率等）
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass  # W&B 记录失败不影响训练

        # 每个 epoch 结束后保存检查点
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)  # 确保目录存在
            state_dict = model.state_dict()  # 获取模型参数
            state_dict['mask_values'] = dataset.mask_values  # 保存掩码类别值（推理时需要）
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))  # 保存到文件
            logging.info(f'Checkpoint {epoch} saved!')  # 记录保存信息


def get_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')  # 训练轮数，默认 5
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')  # 批次大小，默认 1
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')  # 学习率，默认 1e-5
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')  # 加载预训练模型路径
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')  # 图像缩放因子，默认 0.5
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')  # 验证集占比 (0-100)，默认 10%
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')  # 启用混合精度
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')  # 使用双线性上采样
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')  # 类别数，默认 2

    return parser.parse_args()  # 解析并返回参数


if __name__ == '__main__':
    args = get_args()  # 获取命令行参数

    # 配置日志格式
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # 选择设备：有 GPU 用 cuda，否则用 cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')  # 记录使用的设备

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # 创建 U-Net 模型：3 通道输入（RGB），类别数由参数指定，上采样方式由参数指定
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # 使用 channels_last 内存格式，在 CUDA 上可加速
    model = model.to(memory_format=torch.channels_last)

    # 记录模型结构信息
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:  # 如果指定了预训练模型路径
        state_dict = torch.load(args.load, map_location=device)  # 加载权重
        del state_dict['mask_values']  # 移除 mask_values（非模型参数）
        model.load_state_dict(state_dict)  # 加载到模型
        logging.info(f'Model loaded from {args.load}')  # 记录加载信息

    model.to(device=device)  # 将模型移到目标设备
    try:
        # 尝试训练
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,  # 百分比转小数
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:  # 如果 GPU 显存不足
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()  # 清空 GPU 缓存
        model.use_checkpointing()  # 启用梯度检查点（用时间换空间）
        # 使用检查点后重新训练
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
