import argparse
import subprocess
import matplotlib.pyplot as plt
import os
import time

# 解析命令行参数
parser = argparse.ArgumentParser(description='Run two plotting scripts and combine their outputs.')
parser.add_argument('log_file', type=str, help='Path to the log file.')
parser.add_argument('--max_points', type=int, help='Maximum number of data points for the second plot.')
args = parser.parse_args()

# 运行第1个脚本 plt.py 并传递额外的参数
plt_cmd_2 = ['python', 'plt.py', args.log_file]
if args.max_points:
    plt_cmd_2.extend(['--max_points', str(args.max_points)])
subprocess.run(plt_cmd_2, check=True)

# 运行第2个脚本 plt2.py
plt_cmd_1 = ['python', 'plt2.py', args.log_file]
subprocess.run(plt_cmd_1, check=True)

# 假设两个脚本都生成了名为 'output.png' 的图像文件
image_files = ['training_ap.png', 'training_loss.jpg']

# 检查图像文件是否存在
for img_file in image_files:
    if not os.path.exists(img_file):
        print(f"Image file {img_file} not found.")
        exit(1)

# 读取图像数据
image1 = plt.imread(image_files[0])
image2 = plt.imread(image_files[1])

# 创建画布和轴
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))  # 1行2列

# 将图像显示在轴上
ax1.imshow(image1)
ax1.axis('off')  # 隐藏坐标轴
ax1.set_title('First Plot')

ax2.imshow(image2)
ax2.axis('off')  # 隐藏坐标轴
ax2.set_title('Second Plot')

# 调整子图间距
plt.tight_layout()

# 保存合并后的图表
plt.savefig('training_loss_and_ap.png', format='png')
plt.close()

print("The combined plot has been saved as 'training_loss_and_ap.png'.")
