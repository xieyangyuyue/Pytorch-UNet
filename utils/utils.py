# 导入 matplotlib 的 pyplot 模块，用于可视化绘图
import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    # 将输入图像和预测掩码并排可视化
    # img:  输入图像（numpy 数组或 PIL 图像）
    # mask: 预测掩码（numpy 数组，每个像素值为类别索引）
    classes = mask.max() + 1  # 类别数 = 掩码最大值 + 1（因为类别从 0 开始）
    fig, ax = plt.subplots(1, classes + 1)  # 创建 1 行 (类别数+1) 列的子图
    ax[0].set_title('Input image')  # 第一个子图标题
    ax[0].imshow(img)               # 第一个子图显示原始输入图像
    # 为每个类别创建一个子图
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')  # 子图标题标注类别编号
        ax[i + 1].imshow(mask == i)  # 显示该类别的二值掩码（当前类别=白色，其他=黑色）
    plt.xticks([]), plt.yticks([])  # 隐藏所有子图的刻度
    plt.show()  # 显示图像窗口
