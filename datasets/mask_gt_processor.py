import os
import cv2
import numpy as np

def update_instance_id(png_path, mask_dir, output_path, start_new_id=1000):
    """
    更新16位PNG灰度图的Instance ID，基于黑白JPG mask文件。

    Args:
        png_path (str): 原始16位PNG灰度图路径。
        mask_dir (str): 存放黑白JPG mask文件的目录路径。
        output_path (str): 更新后的PNG灰度图保存路径。
        start_new_id (int): 新的Instance ID起始值。
    """
    # 读取原始16位PNG灰度图
    original_img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if original_img is None:
        raise FileNotFoundError(f"无法读取文件: {png_path}")

    # 确保原始图像是16位
    if original_img.dtype != np.uint16:
        raise ValueError("输入的PNG图像不是16位灰度图。")

    # 初始化新ID
    new_id = start_new_id

    # 遍历mask目录中的所有JPG文件
    for mask_file in sorted(os.listdir(mask_dir)):
        if mask_file.lower().endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_file)

            # 读取黑白JPG mask文件
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"警告: 无法读取mask文件: {mask_path}")
                continue

            # 确保mask是二值化的
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # 将mask区域赋予新的Instance ID
            original_img[binary_mask == 0] = new_id

            # 更新新ID
            new_id += 1

    # 保存更新后的灰度图
    cv2.imwrite(output_path, original_img)
    print(f"更新后的灰度图已保存到: {output_path}")

def visualize_colored_instance(image_path, save_path):
    """
    可视化更新后的灰度图，随机赋色并保存为彩色图像。

    Args:
        image_path (str): 更新后的16位PNG灰度图路径。
        save_path (str): 保存彩色图像的路径。
    """
    # 读取更新后的16位灰度图
    gray_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if gray_img is None:
        raise FileNotFoundError(f"无法读取文件: {image_path}")

    # 确保图像是16位
    if gray_img.dtype != np.uint16:
        raise ValueError("输入的PNG图像不是16位灰度图。")

    # 获取唯一的Instance ID
    unique_ids = np.unique(gray_img)
    unique_ids = unique_ids[unique_ids != 0]  # 排除背景ID (0)

    # 创建随机颜色映射
    color_map = {uid: np.random.randint(0, 256, size=3, dtype=np.uint8) for uid in unique_ids}

    # 创建彩色图像
    color_img = np.zeros((*gray_img.shape, 3), dtype=np.uint8)
    for uid, color in color_map.items():
        color_img[gray_img == uid] = color

    # 保存彩色图像
    cv2.imwrite(save_path, color_img)
    print(f"彩色图像已保存到: {save_path}")


if __name__ == "__main__":
    # 示例用法
    path = '/home/rylynn/Downloads/mask_gt_processor'
    png_path = os.path.join(path, 'back_010_seg_uint16.png') # 替换为原始16位PNG灰度图路径
    mask_dir = os.path.join(path, 'correct_mask')      # 替换为存放黑白JPG mask文件的目录路径
    output_path = os.path.join(path, 'back_010_seg_uint16_output.png') # 替换为输出的PNG灰度图路径
    viz_path = os.path.join(path, 'back_010_seg_output.png') # 替换为输出的PNG灰度图路径

    update_instance_id(png_path, mask_dir, output_path, start_new_id=5700)
    visualize_colored_instance(output_path, viz_path)

