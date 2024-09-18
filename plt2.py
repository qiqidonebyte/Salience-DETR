import sys
import re
import matplotlib.pyplot as plt


def convert_time_to_hours(time_str):
    # 检查字符串中是否包含 "day"
    if "day" in time_str:
        # 使用正则表达式匹配天数和时间
        match = re.search(r"(\d+)\s*day,\s*(\d{2}:\d{2}:\d{2})", time_str)
        if match:
            # 提取天数和时间
            days = int(match.group(1))
            hours, minutes, seconds = map(int, match.group(2).split(':'))

            # 将天数转换为小时并加上原有的小时数
            total_hours = days * 24 + hours

            # 格式化为新的字符串
            new_time_str = f"{total_hours:02}:{minutes:02}:{seconds:02}"
            return new_time_str
        else:
            return "Invalid time format"
    else:
        # 如果没有 "day"，直接返回原始时间字符串
        return time_str


def parse_ap_values(file_path):
    # 正则表达式，用于匹配日志文件中的AP值
    ap_regex = re.compile(
        r'\(AP\) @\[ IoU=0.50:0.95 \s*\| \s*area=\s*(all| small|medium| large) \s*\| \s*maxDets=100 \] \s*=\s*([0-9]+\.\d+)')
    # 正则表达式匹配行末的数字
    ap_mean_regx = re.compile(r'\|\s*mean\s+results.*?\|', re.DOTALL)
    # 训练时讲
    time_regx = re.compile(r"Training time:\s*(\d+\s*day,\s*\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2})")

    # 初始化字典来存储AP值
    ap_values = {'all': [], ' small': [], 'medium': [], ' large': [], 'mean': []}
    # epoch_number
    origin_epoch = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # for config
                match = re.search(r'^model_path\s*=\s*(.*)', line)
                if match:
                    config = match.group(1)
                    print("Config:", config)
                # for backbone
                match = re.search(r"Backbone architecture:\s*(\w+)", line)
                if match:
                    backbone_name = match.group(1)
                    print("Backbone:", backbone_name)
                # for APs
                match = ap_regex.search(line)
                if match:
                    area, value = match.groups()
                    value = float(value)
                    ap_values[area].append(value)
                    print(f"Found AP for {area}: {value}")
                # for ap mean
                match = ap_mean_regx.search(line)
                if match:
                    # 从匹配的字符串中提取所有的数字
                    numbers = re.findall(r'\b\d+\.\d+\b', line)
                    # 获取最后一个数字
                    last_number = float(numbers[-1])
                    ap_values['mean'].append(last_number)
                    print(f"Extracted result AP: {last_number}")
                    print('========')

                # for epoch
                epoch_pattern = re.compile(r'Epoch:\s*(\d+)')
                # 使用正则表达式搜索数字
                match = epoch_pattern.search(line)
                if match:
                    # 提取并打印数字
                    epoch_number = int(match.group(1))  # 将捕获的字符串转换为整数
                    if epoch_number != origin_epoch:
                        origin_epoch = epoch_number
                        print(f"Epoch number: {epoch_number}")
                # 正则匹配训练时间
                match = time_regx.search(line)
                if match:
                    print("Training Time:", convert_time_to_hours(match.group(1)))
                    print("=======")
            return ap_values
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None


def main():
    # 检查命令行参数数量
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_log_file>")
        sys.exit(1)

    # 从命令行参数获取日志文件路径
    log_file_paths = sys.argv[1:]
    all_ap_values = {'all': [], ' small': [], 'medium': [], ' large': [], 'mean': []}
    for file_path in log_file_paths:
        ap_values = parse_ap_values(file_path)
        if ap_values:
            for area, values in ap_values.items():
                all_ap_values[area].extend(values)
    print(all_ap_values)
    # 绘制折线图
    plt.figure(figsize=(12, 6))

    # 检查每个区域是否有AP值，并绘制
    label_dict = {
        'all': 'mAP',  # 'all' 对应 mAP 的名称
        ' small': 'AP_S', 'medium': 'AP_M', ' large': 'AP_L',
        'mean': 'mAP50'  # 'mean' 对应 mAP50 的名称

    }
    for area, values in all_ap_values.items():
        if values:  # 如果该区域有AP值
            plt.plot(values, label=f'AP for {label_dict[area]}')

    plt.xlabel('Epoch')
    plt.ylabel('AP')
    plt.title('AP Values Over Epochs')
    plt.legend()
    plt.grid(True)

    # 保存图表为文件，不显示
    plt.savefig('training_ap.png', format='png')
    plt.close()

    print("The plot has been saved as 'training_ap.png'.")


if __name__ == "__main__":
    main()
