# U-Net：基于 PyTorch 的语义分割

![PyTorch](https://img.shields.io/badge/PyTorch-v1.13+-red.svg?logo=PyTorch&style=for-the-badge)
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge)

基于 [U-Net](https://arxiv.org/abs/1505.04597) 论文的 PyTorch 实现，用于 Kaggle [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) 汽车图像分割。

---

## 目录

- [环境配置](#环境配置)
  - [Docker 方式](#docker-方式)
  - [本地方式](#本地方式)
  - [requirements.txt 说明](#requirementstxt-说明)
- [数据准备](#数据准备)
- [训练](#训练)
  - [训练参数详解](#训练参数详解)
  - [完整训练命令](#完整训练命令)
- [推理](#推理)
  - [predict.py（推荐）](#predictpy推荐)
  - [batch_predict.py（旧版，有 bug）](#batch_predictpy旧版有-bug)
  - [推理参数详解](#推理参数详解)
  - [完整推理命令](#完整推理命令)
- [⚠️ C 序问题（重要）](#️-c-序问题重要)
  - [问题说明](#问题说明)
  - [修复方案](#修复方案)
  - [fix_submission.py 补救工具](#fix_submissionpy-补救工具)
- [项目结构](#项目结构)
- [Weights & Biases](#weights--biases)
- [预训练模型](#预训练模型)

---

## 环境配置

### Docker 方式

Dockerfile 基于 NVIDIA 官方 PyTorch 镜像，已预装 PyTorch + CUDA：

```dockerfile
FROM nvcr.io/nvidia/pytorch:22.11-py3
```

```bash
# 1. 安装 Docker
curl https://get.docker.com | sh && sudo systemctl --now enable docker

# 2. 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 3. 运行容器
sudo docker run --rm --shm-size=8g --ulimit memlock=-1 --gpus all -it milesial/unet

# 4. 在容器内下载数据并训练
bash scripts/download_data.sh
python train.py --amp
```

### 本地方式

```bash
# 1. 安装 CUDA（https://developer.nvidia.com/cuda-downloads）

# 2. 安装 PyTorch（必须手动安装，requirements.txt 中不包含）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 下载数据并训练
bash scripts/download_data.sh
python train.py --amp
```

### requirements.txt 说明

`requirements.txt` 中**不包含** `torch` 和 `torchvision`，因为：

- **Docker 方式**：基础镜像 `nvcr.io/nvidia/pytorch:22.11-py3` 已预装 PyTorch + torchvision + CUDA
- **本地方式**：需要根据你的 CUDA 版本手动安装对应版本的 PyTorch

`requirements.txt` 仅列出额外依赖：

| 依赖 | 版本 | 用途 |
|------|------|------|
| matplotlib | 3.6.2 | 可视化绘图 |
| numpy | 1.23.5 | 数值计算 |
| Pillow | 9.3.0 | 图像读取 |
| tqdm | 4.64.1 | 进度条 |
| wandb | 0.13.5 | 实验跟踪 |

---

## 数据准备

Carvana 数据来自 [Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge/data)。

```bash
bash scripts/download_data.sh
```

数据目录结构：

```
data/
├── imgs/       # 训练图像（RGB .jpg）
│   ├── 00087d6bc417.jpg
│   ├── 0a1234567890.jpg
│   └── ...
└── masks/      # 训练掩码（灰度 .gif）
    ├── 00087d6bc417_mask.gif
    ├── 0a1234567890_mask.gif
    └── ...
```

---

## 训练

### 训练参数详解

| 参数 | 短写 | 类型 | 默认值 | 含义 | 使用建议 |
|------|------|------|--------|------|----------|
| `--epochs` | `-e` | int | `5` | 训练轮数 | Carvana 建议 5-20 |
| `--batch-size` | `-b` | int | `1` | 批次大小 | 显存够就开大（4-16） |
| `--learning-rate` | `-l` | float | `1e-5` | 学习率 | 1e-5 ~ 1e-4 常用 |
| `--load` | `-f` | str | `False` | 加载已有模型继续训练 | 断点续训或微调 |
| `--scale` | `-s` | float | `0.5` | 图像缩放因子 | 0.5=半分辨率（快），1.0=原分辨率（精） |
| `--validation` | `-v` | float | `10.0` | 验证集占比 (0-100) | 默认 10% |
| `--amp` | 无 | flag | `False` | 混合精度 | **强烈建议开启**，省显存+加速 |
| `--bilinear` | 无 | flag | `False` | 双线性上采样 | 省显存但精度略低；不指定则用转置卷积 |
| `--classes` | `-c` | int | `2` | 类别数 | 二分类=2，多分类=类别数 |

### 完整训练命令

```bash
# 基本训练（禁用 W&B 日志）
WANDB_MODE=disabled python train.py \
    --epochs 10 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --scale 0.5 \
    --validation 10 \
    --classes 2 \
    --amp \
    --bilinear

# 从检查点继续训练
python train.py \
    --epochs 20 \
    --batch-size 8 \
    --learning-rate 3e-5 \
    --scale 0.5 \
    --classes 2 \
    --amp \
    --load checkpoints/checkpoint_epoch10.pth

# 原始分辨率训练（精度最高，需要大显存）
python train.py \
    --epochs 15 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --scale 1.0 \
    --amp

# 多分类训练
python train.py \
    --epochs 10 \
    --batch-size 4 \
    --learning-rate 3e-5 \
    --scale 0.5 \
    --classes 4 \
    --amp \
    --bilinear
```

训练检查点保存在 `checkpoints/checkpoint_epoch{N}.pth`。

---

## 推理

### predict.py（推荐）

`predict.py` 是增强版批量推理脚本，包含五级自检，已修复 C 序问题。

**三步工作流：**

```bash
# 第 1 步：只跑自检，不推理（约 2 分钟，必须先全部 PASS）
python predict.py \
    --model checkpoints/checkpoint_epoch10.pth \
    --test-dir ./data/test_hq \
    --selftest

# 第 2 步：小批量试跑 500 张，先提交验分
python predict.py \
    --model checkpoints/checkpoint_epoch10.pth \
    --test-dir ./data/test_hq \
    --limit 500 \
    --output submission_500.csv

# 第 3 步：全量推理
python predict.py \
    --model checkpoints/checkpoint_epoch10.pth \
    --test-dir ./data/test_hq \
    --output submission.csv \
    --batch-size 16 \
    --scale 0.5 \
    --classes 2
```

**从 zip 推理：**

```bash
python predict.py \
    --model best_model.pth \
    --test-zip /path/to/test.zip \
    --output submission.csv \
    --batch-size 16 \
    --scale 0.5
```

### batch_predict.py（旧版，有 bug）

⚠️ **此脚本存在 C 序 bug（`order='F'`），不建议直接使用。** 详见 [C 序问题](#️-c-序问题重要)。

```bash
# 如果仍要使用（注意结果需要用 fix_submission.py 修正）
python batch_predict.py \
    --model best_model.pth \
    --test-zip /path/to/test.zip \
    --output submission_raw.csv \
    --batch-size 16 \
    --scale 0.5 \
    --bilinear

# 修正 C 序
python fix_submission.py submission_raw.csv -o submission.csv
```

### 推理参数详解

#### predict.py 参数

| 参数 | 短写 | 类型 | 默认值 | 含义 |
|------|------|------|--------|------|
| `--model` | `-m` | str | **必填** | 模型权重 `.pth` 路径 |
| `--test-dir` | `-d` | str | 无 | 测试图目录 |
| `--test-zip` | `-z` | str | 无 | 测试图 zip（自动预解压） |
| `--output` | `-o` | str | `submission.csv` | 输出 CSV 路径 |
| `--batch-size` | `-b` | int | `16` | 批次大小 |
| `--scale` | `-s` | float | `0.5` | 缩放因子（**必须与训练一致**） |
| `--classes` | `-c` | int | `2` | 类别数 |
| `--no-amp` | 无 | flag | `False` | 关闭混合精度（默认开启） |
| `--workers` | 无 | int | `min(8,cpu)` | 数据加载进程数 |
| `--limit` | 无 | int | `0` | 只跑前 N 张（试跑用，0=全部） |
| `--selftest` | 无 | flag | `False` | 只跑五级自检，不推理 |
| `--viz` | 无 | int | `0` | 可视化前 N 张 |
| `--mask-dir` | 无 | str | `./data/masks` | 训练掩码目录（自检用） |
| `--train-img-dir` | 无 | str | `./data/imgs` | 训练图目录（自检用） |
| `--official-csv` | 无 | str | `None` | 官方 train_masks.csv(.zip) |
| `--sample-sub` | 无 | str | `None` | 官方 sample_submission.csv |

#### batch_predict.py 参数

| 参数 | 短写 | 类型 | 默认值 | 含义 |
|------|------|------|--------|------|
| `--model` | `-m` | str | **必填** | 模型权重路径 |
| `--test-dir` | `-d` | str | 无 | 测试图目录 |
| `--test-zip` | `-z` | str | 无 | 测试图 zip |
| `--output` | `-o` | str | `submission.csv` | 输出 CSV |
| `--batch-size` | `-b` | int | `32` | 批次大小 |
| `--scale` | `-s` | float | `0.5` | 缩放因子 |
| `--classes` | `-c` | int | `2` | 类别数 |
| `--bilinear` | 无 | flag | `False` | 双线性上采样 |
| `--amp` | 无 | flag | `True` | 混合精度（⚠️ 有 bug，无法关闭） |

### 完整推理命令

```bash
# 推荐：使用 predict.py（C 序正确，含自检）
python predict.py \
    --model checkpoints/checkpoint_epoch10.pth \
    --test-zip /run/media/xie/软件/test.zip \
    --output submission.csv \
    --batch-size 16 \
    --scale 0.5 \
    --classes 2

# 带官方 CSV 比对的自检模式
python predict.py \
    --model checkpoints/checkpoint_epoch10.pth \
    --test-dir ./data/imgs \
    --selftest \
    --official-csv ./data/train_masks.csv \
    --mask-dir ./data/masks \
    --viz 4
```

---

## ⚠️ C 序问题（重要）

### 问题说明

Carvana 比赛官方评估端在解码 RLE 时使用 **C 序（行优先，row-major）**：

```python
mask = flat_array.reshape(H, W)  # numpy 默认 order='C'
```

| 文件 | RLE 编码顺序 | 状态 | 影响 |
|------|-------------|------|------|
| `batch_predict.py` | `order='F'`（列优先） | ❌ **错误** | 提交 Dice ≈ 0.35（几乎全错） |
| `predict.py` | `order='C'`（行优先） | ✅ **正确** | 与官方一致，Dice ≈ 0.97 |

**原因**：`batch_predict.py` 用列优先展平掩码再编码，官方用行优先解码，行列错位导致掩码被"转置"。

**batch_predict.py 还有一个 bug**：`--amp` 参数使用 `action='store_true'` + `default=True`，导致混合精度永远开启、无法关闭。`predict.py` 已修复为 `--no-amp` 反向开关。

### 修复方案

1. **推荐**：直接使用 `predict.py`（已修复 C 序 + AMP bug，含五级自检）
2. **手动修**：将 `batch_predict.py` 中 `rle_encode` 的 `order='F'` 改为 `order='C'`
3. **补救**：如果已经用 `batch_predict.py` 跑完了全量推理，用 `fix_submission.py` 无损转换

### fix_submission.py 补救工具

无需重跑推理，几分钟即可将 F 序 CSV 转为正确的 C 序：

```bash
# 自动检测顺序并转换
python fix_submission.py submission_raw.csv -o submission_fixed.csv

# 强制指定原顺序为 F
python fix_submission.py submission_raw.csv --from-order F -o submission_fixed.csv

# 只诊断不转换
python fix_submission.py submission_raw.csv --check-only
```

**原理**：全程只是下标置换（F 序位置 → C 序位置），零信息损失。

---

## 项目结构

```
Pytorch-UNet/
├── train.py              # 训练脚本
├── predict.py            # 批量推理 + 五级自检（推荐使用）
├── batch_predict.py      # 批量推理旧版（⚠️ C 序 bug）
├── fix_submission.py     # F 序 → C 序无损转换工具
├── evaluate.py           # 验证集评估（Dice 系数）
├── hubconf.py            # torch.hub 预训练模型加载
├── requirements.txt      # 依赖列表（不含 torch）
├── Dockerfile            # Docker 构建文件
├── unet/                 # U-Net 模型定义
│   ├── __init__.py       # 包初始化
│   ├── unet_model.py     # 完整 U-Net 网络
│   └── unet_parts.py     # 网络组件（DoubleConv/Down/Up/OutConv）
├── utils/                # 工具模块
│   ├── data_loading.py   # 数据加载与预处理
│   ├── dice_score.py     # Dice 系数与损失
│   └── utils.py          # 可视化工具
├── scripts/
│   └── download_data.sh  # Kaggle 数据下载脚本
└── data/                 # 数据目录（需自行下载）
    ├── imgs/             # 训练图像
    └── masks/            # 训练掩码
```

---

## Weights & Biases

训练过程可通过 [W&B](https://wandb.ai/) 实时可视化：损失曲线、验证曲线、权重/梯度直方图、预测掩码。

```bash
# 禁用 W&B（离线训练）
WANDB_MODE=disabled python train.py --amp

# 启用 W&B（会打印 dashboard 链接）
python train.py --amp

# 使用已有 W&B 账号
export WANDB_API_KEY=your_key
python train.py --amp
```

---

## 预训练模型

Carvana 数据集预训练模型可通过 torch.hub 加载：

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
```

可用缩放比例：`0.5` 和 `1.0`。

---

## 原始论文

Olaf Ronneberger, Philipp Fischer, Thomas Brox:
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
