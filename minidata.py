import os
import shutil
import json
import argparse

def create_new_dataset(original_dataset_dir, new_dataset_dir, selection_interval, subsets):
    for subset in subsets:
        # 创建新的图片和标注文件目录
        new_images_dir = os.path.join(new_dataset_dir, f'{subset}2017')
        new_annotations_dir = os.path.join(new_dataset_dir, 'annotations')
        os.makedirs(new_images_dir, exist_ok=True)
        os.makedirs(new_annotations_dir, exist_ok=True)
        # 定义原始和新的图片目录
        original_images_dir = os.path.join(original_dataset_dir, f'{subset}2017')
        new_images_dir = os.path.join(new_dataset_dir, f'{subset}2017')
        os.makedirs(new_images_dir, exist_ok=True)

        # 定义原始和新的标注文件路径
        original_annotations_file = os.path.join(original_dataset_dir, 'annotations', f'instances_{subset}2017.json')
        new_annotations_file = os.path.join(new_dataset_dir, 'annotations', f'instances_{subset}2017.json')

        # 读取原始标注文件
        with open(original_annotations_file, 'r') as f:
            annotations = json.load(f)

        # 创建新的标注文件
        new_annotations = {
            'images': [],
            'annotations': [],
            'categories': annotations['categories']
        }

        # 记录新数据集中的图片ID
        new_image_ids = set()

        # 遍历原始图片目录中的所有图片文件
        for file_number, filename in enumerate(sorted(os.listdir(original_images_dir)), start=1):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if file_number % selection_interval == 0:
                    original_image_path = os.path.join(original_images_dir, filename)
                    new_image_path = os.path.join(new_images_dir, filename)
                    shutil.copy2(original_image_path, new_image_path)

                    # 从原始标注中提取图片ID
                    for image in annotations['images']:
                        if image['file_name'] == filename:
                            new_image_ids.add(image['id'])
                            new_annotations['images'].append(image)
                            break

                    # 更新新标注文件中的annotations
                    for annotation in annotations['annotations']:
                        if annotation['image_id'] in new_image_ids:
                            new_annotations['annotations'].append(annotation)

        # 写入新的标注文件
        with open(new_annotations_file, 'w') as f:
            json.dump(new_annotations, f, indent=4)

        print(f'New dataset for {subset}2017 created successfully.')

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Create a new dataset by selecting images at a specified interval for different subsets.')
parser.add_argument('original_dataset_dir', type=str, help='The directory of the original dataset.')
parser.add_argument('new_dataset_dir', type=str, help='The directory to store the new dataset.')
parser.add_argument('selection_interval', type=int, help='The interval for selecting images (e.g., 10 for every 10th image).')
parser.add_argument('subsets', nargs='+', type=str, choices=['train', 'val', 'test'], help='Subsets to process (train, val, test).')

# 解析命令行参数
args = parser.parse_args()

# 调用函数创建新数据集
create_new_dataset(args.original_dataset_dir, args.new_dataset_dir, args.selection_interval, args.subsets)
# 使用
# python script_name.py /path/to/original/coco2017 /path/to/new_dataset 10 train val test