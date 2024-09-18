import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib
matplotlib.use('Agg')

# 解析命令行参数
parser = argparse.ArgumentParser(description='Draw bounding boxes on images and save them with all predictions.')
parser.add_argument('--image_folder', type=str, required=True, help='The folder containing the images.')
parser.add_argument('--output_folder', type=str, required=True, help='The folder to save annotated images.')
parser.add_argument('--json_file', type=str, required=True, help='The JSON file containing predictions.')
args = parser.parse_args()

# 定义颜色映射，将 category_id 映射到颜色
category_colors = {
    1: 'blue',
    2: 'green',
    3: 'red',
    # 添加更多分类和颜色...
}

# 根据 image_id 获取图像路径
def get_image_path(image_id, image_folder):
    for ext in ['jpg', 'png']:
        image_path = os.path.join(image_folder, f'{image_id}.{ext}')  # 格式化image_id为四位数
        if os.path.exists(image_path):
            return image_path
    return None

def filter_high_score_predictions(predictions, min_score_threshold):
    """
    筛选出高于指定分数阈值的预测结果。
    """
    filtered_predictions = []
    for pred in predictions:
        if pred['score'] >= min_score_threshold:
            filtered_predictions.append(pred)
    return filtered_predictions


# 绘制边界框和类别ID
def draw_boxes(image_path, predictions):
    print(f"Drawing boxes for image: {image_path}")
    with Image.open(image_path) as image:
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # 可以在这里设置分数阈值
        min_score_threshold = 0.5  # 例如，筛选出 score 大于或等于 0.5 的预测
        # 筛选预测
        predictions = filter_high_score_predictions(predictions, min_score_threshold)

        for pred in predictions:
            bbox = pred['bbox']
            category_id = pred['category_id']
            score = pred['score']
            x, y, width, height = map(float, bbox)  # 确保坐标是浮点数

            # 根据 category_id 获取颜色，如果未找到则使用黑色
            color = category_colors.get(category_id, 'black')
            # 绘制边界框
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # 标记类别ID和分数
            # ax.text(x, y, f"{category_id}", fontsize=9, verticalalignment='bottom', horizontalalignment='left', color='white', backgroundcolor='blue')

        # 调整图像显示限制
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis('off')

        # 保存图像
        output_path = os.path.join(args.output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_annotated.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'Annotated image saved to {output_path}')

# 筛选前5个image_id对应的预测结果，并为每个image_id绘制所有预测框
def filter_top_five_predictions(predictions):
    # 按image_id组织预测结果
    image_id_to_predictions = {}
    for pred in predictions:
        image_id = pred['image_id']
        if image_id in image_id_to_predictions:
            image_id_to_predictions[image_id].append(pred)
        else:
            image_id_to_predictions[image_id] = [pred]

    # 选择前5个image_id对应的所有预测结果
    top_five_image_ids = sorted(image_id_to_predictions.keys())[:3]
    top_five_predictions = {image_id: image_id_to_predictions[image_id] for image_id in top_five_image_ids}
    return top_five_predictions

# 主函数
def main():
    print(f"Loading predictions from {args.json_file}")
    # 加载JSON文件
    with open(args.json_file, 'r') as f:
        predictions = json.load(f)

    # 筛选前5个image_id的预测结果
    top_five_predictions = filter_top_five_predictions(predictions)
    print(f"Loading 5 top predictions  {top_five_predictions}")

    # 保存标注图像
    for image_id, preds in top_five_predictions.items():
        image_path = get_image_path(image_id, args.image_folder)
        print(f"Loading image_path:  {image_path}")
        if image_path:
            try:
                draw_boxes(image_path, preds)
            except Exception as e:
                print(f'Error saving annotated image: {e}')
            

if __name__ == "__main__":
    main()