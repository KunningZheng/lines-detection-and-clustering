# external
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# internal
from datasets.line3dpp_loader import parse_line_segments
from datasets.mask_processor import split_masks
from clustering.lines_tools import rasterize_lines

def associate_lines_to_masks(cam_dict, merged_mask_path, line3dpp_path, output_path=None, show=False):
    """
    关联2D线段与mask的对应关系
    
    参数:
        cam_dict: 相机信息字典
        merged_mask_path: 合并掩码路径
        line3dpp_path: 3D线段文件路径
        output_path: 可视化输出路径 (可选)
        show: 是否显示可视化结果 (默认False)
    
    返回:
        mask_to_lines: 字典，key为mask_id，value为关联的线段ID列表
    """
    img_name = cam_dict['img_name'].split('/')[-1]
    width, height = int(cam_dict['width']), int(cam_dict['height'])
    
    # 读取当前航片的merged_mask
    mask_path = os.path.join(merged_mask_path, img_name + '.npy')
    merged_mask = np.load(mask_path, allow_pickle=True)
    # 获取所有唯一的mask ID
    mask_unique_ids = np.unique(merged_mask)
    
    # 解析并栅格化2D线段
    lines2d = parse_line_segments(line3dpp_path, cam_dict['id'], width, height)
    raster_lines = rasterize_lines((height, width), lines2d.reshape(-1, 4))
    
    # 初始化线段与mask的关联数组
    lines2d_mask_id = np.ones((lines2d.shape[0], 2), dtype=int) * -1
    
    # 处理每个mask
    for mask_id in mask_unique_ids:
        mask_id = int(mask_id)  # 确保mask_id是整数类型
        # 跳过背景mask
        if mask_id == -1:  
            continue

        # 提取当前mask的布尔数组
        bool_mask = split_masks(merged_mask, mask_id, resize=(width, height))    
        # 选取与mask对应的线段
        lines_in_mask = raster_lines[bool_mask]
        lines2d_id, counts = np.unique(lines_in_mask, return_counts=True)
        
        for line_id, count in zip(lines2d_id, counts):
            # 没有线段的区域被标记为-1
            if line_id == -1:
                continue
            # 保留像素数最多的mask关联
            if lines2d_mask_id[line_id, 1] < count:
                lines2d_mask_id[line_id, 0] = mask_id
                lines2d_mask_id[line_id, 1] = count
    
    # 构建mask到线段的映射字典
    mask_to_lines = {}
    for line_id, (mask_id, _) in enumerate(lines2d_mask_id):
        mask_id = int(mask_id)  # 确保mask_id是整数类型
        if mask_id != -1:
            mask_to_lines.setdefault(mask_id, []).append(line_id)

    # 构建线段到mask的映射
    line_to_mask = {}
    for line_id, (mask_id, count) in enumerate(lines2d_mask_id):
        mask_id = int(mask_id)  # 确保mask_id是整数类型
        if mask_id != -1:  # 移除没有mask关联的线段
            line_to_mask[line_id] = mask_id

    # 显示或保存结果
    if show:
        merged_mask_resized = cv2.resize(merged_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        visualize_img = visualize_masked_lines((height, width), lines2d, lines2d_mask_id, merged_mask_resized)
        plt.figure(figsize=(10, 10), dpi=100)
        plt.imshow(visualize_img)
    
    if output_path:
        merged_mask_resized = cv2.resize(merged_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        visualize_img = visualize_masked_lines((height, width), lines2d, lines2d_mask_id, merged_mask_resized)
        filename = f"{img_name}_mask_association.png"
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_path, filename), cv2.cvtColor(visualize_img, cv2.COLOR_RGB2BGR))

            
    return mask_to_lines, line_to_mask


def visualize_masked_lines(image_shape, lines2d, lines2d_mask_id, merged_mask):
    """
    可视化线段及其对应的mask区域（半透明填充+线段）
    :param image_shape: 图像尺寸 (height, width)
    :param lines2d: 2D线段数组，形状为(N,4)，每行为[x1,y1,x2,y2]
    :param lines2d_mask_id: 线段对应的mask ID数组，形状为(N,2)，第一列为mask ID
    :param merged_mask: 合并后的mask数组，形状为(height, width)
    :return: 可视化图像 (RGB格式)
    """
    height, width = image_shape
    # 创建浅灰色背景 [200,200,200]
    img = np.full((height, width, 3), [200, 200, 200], dtype=np.uint8)
    
    # 获取所有有效mask ID（排除-1）
    unique_mask_ids = np.unique(lines2d_mask_id[:, 0])
    unique_mask_ids = unique_mask_ids[unique_mask_ids != -1]
    
    # 为每个mask ID生成唯一颜色
    color_dict = {}
    mask_alpha = 0.3  # mask填充透明度
    used_colors = {tuple([200, 200, 200])}  # 禁止使用背景色
    
    for mask_id in unique_mask_ids:
        while True:
            # 生成随机颜色 (BGR格式)
            color = [random.randint(0, 255) for _ in range(3)]
            # 转换为RGB用于颜色检查
            color_rgb = tuple(color[::-1])
            
            # 检查颜色是否唯一且不与背景色接近
            color_diff = np.abs(np.array(color) - [200, 200, 200])
            if color_rgb not in used_colors and np.all(color_diff > 50):
                used_colors.add(color_rgb)
                color_dict[mask_id] = color  # 存储BGR格式颜色
                break
    
    # 先绘制mask区域（半透明填充）
    mask_layer = img.copy()
    for mask_id in color_dict.keys():
        # 创建当前mask的布尔掩码
        mask_area = (merged_mask == mask_id)
        # 用对应颜色填充mask区域
        mask_layer[mask_area] = color_dict[mask_id]
    # 将mask层与原图混合（alpha混合）
    img = cv2.addWeighted(img, 1 - mask_alpha, mask_layer, mask_alpha, 0)
    
    # 再绘制所有线段（不透明）
    for i, line in enumerate(lines2d):
        mask_id = lines2d_mask_id[i, 0]
        if mask_id not in color_dict:
            continue  # 跳过未分配mask
        
        # 注意坐标顺序转换 (x,y格式)
        pt1 = (int(round(line[1])), int(round(line[0])))
        pt2 = (int(round(line[3])), int(round(line[2])))
        cv2.line(img, pt1, pt2, color_dict[mask_id], thickness=2)
    
    return img


def load_mask_lines_association(camerasInfo, merged_mask_path, line3dpp_path, intermediate_output_path):
    """
    加载或计算所有相机的 mask 和 lines 关联数据
    参数:
        camerasInfo: 相机信息列表
        merged_mask_path: 合并掩码路径
        line3dpp_path: 3D线段文件路径
        intermediate_output_path: 中间输出路径，用于存储结果
    返回:
        all_mask_to_lines: 字典，key为相机ID，value为该相机的mask_to_lines关联数据
        all_line_to_mask: 字典，key为相机ID，value为该相机的line_to_mask关联数据
    """
    all_mask_to_lines_path = os.path.join(intermediate_output_path, 'all_mask_to_lines.json')
    all_line_to_mask_path = os.path.join(intermediate_output_path, 'all_line_to_mask.json')
    
    # 如果已经存在所有mask与线段的关联数据，则直接加载
    if os.path.exists(all_mask_to_lines_path) and os.path.exists(all_line_to_mask_path):
        print(f"Loading existing mask and lines associations")
        with open(all_mask_to_lines_path, 'r') as f:
            all_mask_to_lines = json.load(f)
        with open(all_line_to_mask_path, 'r') as f:
            all_line_to_mask = json.load(f)
    # 如果不存在，则计算所有相机的mask与线段的关联
    else:
        all_mask_to_lines = {}
        all_line_to_mask = {}
        for cam_dict in tqdm(camerasInfo, desc="Computing mask to lines associations"):
            mask_to_lines, line_to_mask = associate_lines_to_masks(
                cam_dict, 
                merged_mask_path, 
                line3dpp_path,
                output_path=os.path.join(intermediate_output_path, 'associate_lines_to_masks')
            )
            # 记录所有相机的结果
            all_mask_to_lines[cam_dict['id']] = mask_to_lines
            all_line_to_mask[cam_dict['id']] = line_to_mask
        # 保存所有mask与线段的关联
        with open(all_mask_to_lines_path, 'w') as f:
            json.dump(all_mask_to_lines, f, indent=4)
        with open(all_line_to_mask_path, 'w') as f:
            json.dump(all_line_to_mask, f, indent=4)
    return all_mask_to_lines, all_line_to_mask
