import argparse
import matplotlib.pyplot as plt
import re, os
import numpy as np

# 定义命令行参数解析
parser = argparse.ArgumentParser(description='Plot loss curve from log file.')
parser.add_argument('log_file', type=str, help='Path to the log file.')
parser.add_argument('--max_points', type=int, default=100, help='Maximum number of points to keep on the plot.')

# 解析命令行参数
args = parser.parse_args()

# 初始化存储loss值的列表
loss_values = []

# 正则表达式，用于匹配格式为 'loss: X.XXXX' 的字符串
loss_pattern = re.compile(r'(?i)loss: ([0-9.]+)')

# 读取命令行指定的日志文件并提取loss值
with open(args.log_file, 'r') as file:
    for line in file:
        # 对于每一行，搜索第一个匹配的loss值
        match = loss_pattern.search(line)
        if match:
            # 如果找到，提取并转换为float类型，然后添加到列表中
            loss_value = float(match.group(1))
            loss_values.append(loss_value)

# 去掉前3个点
loss_values = loss_values[5:] if len(loss_values) > 50 else loss_values
# 如果点的数量超过了最大限制，计算采样间隔并进行采样
if len(loss_values) > args.max_points:
    sample_interval = len(loss_values) // args.max_points
    sampled_indices = np.arange(0, len(loss_values), step=sample_interval)
    loss_values = [loss_values[i] for i in sampled_indices]

# 找到最小值及其索引
min_loss = min(loss_values)
min_loss_idx = np.where(np.array(loss_values) == min_loss)[0][0]

# 检查是否有loss值被提取
if not loss_values:
    print("No loss values found in the log file.")
else:
    # 绘制loss曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss', marker='o')  # 使用'o'标记每个数据点
    plt.title('Training Loss Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 计算x坐标位置，通常我们使用最小值点的索引+1，防止标注在轴上
    x_position = min_loss_idx + 1 if min_loss_idx < len(loss_values) - 1 else min_loss_idx

    # 在图上标注最小值
    plt.annotate(f'Min Loss: {min_loss:.2f}', xy=(x_position, min_loss), xytext=(0, 30),
                 textcoords='offset points', ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.2),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

    # 保存图像前删除旧图像
    output_filename = 'training_loss.jpg'
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # 保存图像到文件，格式为JPEG
    plt.savefig(output_filename, format='jpeg')
    # 关闭图形，释放资源
    plt.close()
    print(f'{output_filename} saved successfully.')
