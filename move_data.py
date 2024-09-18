import os
import shutil

# 设置源目录和目标目录
src_dir = '/home/rjzy/Documents/SalienceDETR/Salience-DETR/data/coco/test2017/data'
dest_dir = '/home/rjzy/Documents/SalienceDETR/Salience-DETR/data/coco/test2017'

# 定义每批移动的文件数量
batch_size = 1000

# 确保目标目录存在
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 获取源目录下的所有文件
files_to_move = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# 计算需要的批次数量
total_files = len(files_to_move)
batches = (total_files + batch_size - 1) // batch_size

# 分批移动文件
for i in range(batches):
    print(f'Moving batch {i + 1} of {batches}')
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, total_files)
    batch_files = files_to_move[start_index:end_index]

    # 移动一批文件
    for file_name in batch_files:
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        shutil.move(src_file, dest_file)

    print(f'Batch {i + 1} completed.')