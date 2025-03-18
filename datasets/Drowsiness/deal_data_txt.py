import glob
import os
import shutil

import cv2
import tqdm


def draw_bboxes_from_labels(image_folder, label_folder, output_folder):
    """
    从标签文件中读取边界框信息，并在对应的图像上绘制边界框和类别。

    Args:
        image_folder (str): 包含图像文件的文件夹路径。
        label_folder (str): 包含标签文件的文件夹路径。
        output_folder (str): 保存标注后图像的输出文件夹路径。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    for label_file in tqdm.tqdm(label_files, desc='Processing'):
        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)
        output_image_path = os.path.join(output_folder, image_file)

        if not os.path.exists(image_path):
            print(f"Image file {image_file} not found.")
            continue

        # 读取图像
        image = cv2.imread(image_path)

        # 读取对应的标签文件并绘制边界框
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                category, x_center, y_center, width, height = map(float, parts)
                x_center, y_center, width, height = x_center * image.shape[1], y_center * image.shape[0], width * \
                                                    image.shape[1], height * image.shape[0]
                x_min, y_min = int(x_center - width / 2), int(y_center - height / 2)
                x_max, y_max = int(x_center + width / 2), int(y_center + height / 2)

                # 绘制边界框
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # 标注类别
                cv2.putText(image, str(int(category)), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                            2)

        # 保存标注后的图像
        cv2.imwrite(output_image_path, image)


def ensure_clean_directory(directory_path):
    """
    确保目录不存在，如果存在，则删除并重新创建空目录。

    Args:
        directory_path (str): 要检查和清理的目录路径。
    """
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)  # 删除存在的目录及其所有内容
    os.makedirs(directory_path)  # 创建新的空目录


def read_polygon_labels(file_path):
    """
    从文件中读取多边形标签。

    Args:
        file_path (str): 标签文件的路径。

    Returns:
        list of tuples: 每个元组包含类别索引和多边形的(x, y)坐标列表。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    polygons = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:  # 确保数值总数为奇数且至少为3
            continue  # 如果不符合条件，则跳过当前行
        category = int(parts[0])
        points = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
        polygons.append((category, points))
    return polygons


def polygon_to_bbox(polygon):
    """
    将多边形坐标转换为边界框坐标。

    Args:
        polygon (list of tuples): 多边形的坐标列表，每个元组包含(x, y)坐标。

    Returns:
        tuple: 边界框的坐标(x_min, y_min, x_max, y_max)。
    """
    xs = [coord[0] for coord in polygon]
    ys = [coord[1] for coord in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def convert_labels(input_dirs, output_dirs):
    """
    转换包含多边形分割标注的标签文件为边界框坐标形式的标签文件。

    Args:
        input_dirs (list): 输入目录列表。
        output_dirs (list): 输出目录列表。
    """
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        ensure_clean_directory(output_dir)  # 确保输出目录是干净的
        for label_path in glob.glob(os.path.join(input_dir, '*.txt')):
            polygons = read_polygon_labels(label_path)
            if polygons:
                output_path = os.path.join(output_dir, os.path.basename(label_path))
                with open(output_path, 'w') as f:
                    for category, polygon in polygons:
                        x_min, y_min, x_max, y_max = polygon_to_bbox(polygon)
                        # 计算边界框的中心点坐标和尺寸
                        bbox_center_x = (x_min + x_max) / 2
                        bbox_center_y = (y_min + y_max) / 2
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min
                        bbox_line = f"{category} {bbox_center_x} {bbox_center_y} {bbox_width} {bbox_height}\n"
                        f.write(bbox_line)
                print(f"Converted and saved: {output_path}")


# # 指定输入和输出目录
# input_dirs = ['train/labels', 'test/labels', 'valid/labels']
# output_dirs = ['train_converted/labels', 'test_converted/labels', 'valid_converted/labels']
#
# # 执行转换
# convert_labels(input_dirs, output_dirs)

# 指定图像文件夹、标签文件夹和输出文件夹的路径
image_folder = 'train/images'
label_folder = 'train_converted/labels'
output_folder = 'train_converted/images'

# 执行绘制和保存
draw_bboxes_from_labels(image_folder, label_folder, output_folder)
