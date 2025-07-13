import os
import cv2
import numpy as np
from tqdm import tqdm  # 用于进度条显示
import random
import matplotlib.pyplot as plt


def visualize_merged_mask(merged_mask, output_path=None, show=True, alpha=0.7, dpi=100, colormap='random'):
    """
    可视化合并后的掩码，为每个实例ID分配不同颜色，固定背景色为浅灰色
    
    参数:
        merged_mask: 合并后的掩码数组 (2D numpy数组)
        output_path: 保存图像的文件路径 (可选)
        show: 是否显示图像 (默认True)
        alpha: 掩码透明度 (0-1, 默认0.7)
        dpi: 图像分辨率 (默认100)
        colormap: 使用的颜色映射 ('random', 'tab20', 'viridis'等, 默认'random')
    """
    # 获取唯一ID（排除背景-1）
    unique_ids = np.unique(merged_mask)
    unique_ids = unique_ids[(unique_ids != -1)]
    
    # 创建彩色图像
    if colormap == 'random':
        # 固定背景色（使用不太可能随机生成的浅灰色）
        BACKGROUND_COLOR = [200, 200, 200]
        
        # 为每个ID生成确定性颜色（基于ID的哈希值）
        color_dict = {-1: BACKGROUND_COLOR}  # 固定背景色
        
        for id_val in unique_ids:
            # 使用ID的哈希值生成确定性颜色
            random.seed(id_val)  # 固定随机种子
            while True:
                color = [random.randint(50, 255), 
                        random.randint(50, 255), 
                        random.randint(50, 255)]
                # 确保颜色不与背景色冲突
                if color != BACKGROUND_COLOR:
                    color_dict[id_val] = color
                    break
        
        # 创建RGB图像
        color_mask = np.zeros(merged_mask.shape + (3,), dtype=np.uint8)
        for id_val, color in color_dict.items():
            color_mask[merged_mask == id_val] = color
    
    else:
        # 使用matplotlib颜色映射
        cmap = plt.get_cmap(colormap)
        normalized_ids = merged_mask.copy().astype(float)
        
        # 将背景设为NaN（-1设为背景）
        normalized_ids[(merged_mask == -1)] = np.nan
        
        # 归一化ID到0-1范围
        if len(unique_ids) > 0:
            min_id, max_id = np.min(unique_ids), np.max(unique_ids)
            normalized_ids = (normalized_ids - min_id) / (max_id - min_id)
        
        # 应用颜色映射
        color_mask = cmap(normalized_ids)[:, :, :3]
        color_mask = (color_mask * 255).astype(np.uint8)
        # 背景设为固定浅灰色
        BACKGROUND_COLOR = [200, 200, 200]
        color_mask[(merged_mask == -1)] = BACKGROUND_COLOR
    
    # 创建带有透明度的可视化
    background = np.ones_like(color_mask) * 255  # 白色背景
    blended = (background * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    
    # 添加边界轮廓（不包括背景ID=-1）
    contours_img = blended.copy()
    for id_val in unique_ids:
        mask = (merged_mask == id_val).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_img, contours, -1, (0, 0, 0), 1)  # 黑色轮廓
    
    # 显示或保存结果
    if show:
        plt.figure(figsize=(10, 10), dpi=dpi)
        plt.imshow(contours_img)
        plt.title(f"Instance Segmentation - {len(unique_ids)} objects")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    if output_path:
        # 在文件名中添加对象数量
        base, ext = os.path.splitext(output_path)
        output_path_with_count = f"{base}_{len(unique_ids)}objs{ext}"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path_with_count, cv2.cvtColor(contours_img, cv2.COLOR_RGB2BGR))
    
    return contours_img


def merge_masks_to_npy(sam_mask_path, output_dir, overlap_threshold=0.5):
    """
    将每个文件夹内的mask合并为一个NPY文件，使用新的合并策略：
    如果小掩码的50%以上的区域在大掩码的范围内，则两个掩码合并
    
    参数:
        sam_mask_path: 包含mask文件夹的根目录
        output_dir: 输出NPY文件的目录
        overlap_threshold: 重叠比例阈值(默认0.5)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有文件夹并添加进度条
    folders = [f for f in os.listdir(sam_mask_path) if os.path.isdir(os.path.join(sam_mask_path, f))]
    for folder_name in tqdm(folders, desc="Processing mask folders"):
        folder_path = os.path.join(sam_mask_path, folder_name)
        output_path = os.path.join(output_dir, f"{folder_name}.npy")

        # 检查是否已处理过
        if os.path.exists(output_path):
            continue
        
        mask_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        if not mask_files:
            continue
            
        # 收集所有掩码及其ID和面积
        masks_info = []
        for mask_file in mask_files:
            mask_path = os.path.join(folder_path, mask_file)
            mask_id = int(mask_file.split('.')[0])  # 文件名格式为"mask_id.png"
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # 二值化处理：任何非零值视为前景
            _, binary_mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)
            
            # 计算掩码面积（非零像素数量）
            area = np.count_nonzero(binary_mask)
            masks_info.append((mask_id, binary_mask, area))
        
        # 按面积从大到小排序
        masks_info.sort(key=lambda x: x[2], reverse=True)
        
        # 初始化合并后的掩码，背景设为-1
        merged_mask = np.full(masks_info[0][1].shape, fill_value=-1, dtype=np.int16)
        
        # 创建已处理掩码的列表
        processed_masks = []
        
        # 按面积从大到小处理掩码
        for i, (mask_id, binary_mask, area) in enumerate(masks_info):
            # 检查当前掩码与已合并掩码的重叠情况
            merged_areas = {}
            
            # 查找所有与当前掩码重叠的已处理掩码
            for processed_id, processed_mask in processed_masks:
                overlap = np.logical_and(binary_mask, merged_mask == processed_id)
                overlap_area = np.count_nonzero(overlap)
                
                if overlap_area > 0:
                    # 计算重叠比例（相对于当前小掩码的面积）
                    overlap_ratio = overlap_area / area
                    if overlap_ratio >= overlap_threshold:
                        # 如果重叠比例超过阈值，记录重叠区域和对应的已处理掩码ID
                        merged_areas[processed_id] = overlap
            
            if merged_areas:
                # 如果有满足条件的重叠，选择重叠面积最大的掩码进行合并
                best_match_id = max(merged_areas.keys(), key=lambda x: merged_areas[x].sum())
                
                # 合并到最佳匹配的掩码中
                apply_mask = (binary_mask == 1)
                merged_mask[apply_mask] = best_match_id
            else:
                # 没有满足条件的重叠，作为新掩码添加
                apply_mask = (binary_mask == 1) & (merged_mask == -1)
                merged_mask[apply_mask] = mask_id
            
            # 将当前掩码信息添加到已处理列表中
            processed_masks.append((mask_id if not merged_areas else best_match_id, binary_mask))
        
        # 保存为NPY文件
        np.save(output_path, merged_mask)

        # 可视化合并后的掩码
        visualize_path = os.path.join(output_dir, 'visualization')
        os.makedirs(visualize_path, exist_ok=True)
        visualize_merged_mask(merged_mask, output_path=os.path.join(visualize_path, f"{folder_name}_vis.png"), show=False)


def cover_masks_to_npy(sam_mask_path, output_dir):
    """
    将每个文件夹内的mask合并为一个NPY文件，使用"大掩码覆盖小掩码"策略
    
    参数:
        sam_mask_path: 包含mask文件夹的根目录
        output_dir: 输出NPY文件的目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有文件夹并添加进度条
    folders = [f for f in os.listdir(sam_mask_path) if os.path.isdir(os.path.join(sam_mask_path, f))]
    for folder_name in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(sam_mask_path, folder_name)
        output_path = os.path.join(output_dir, f"{folder_name}.npy")
        
        # 检查是否已处理过
        if os.path.exists(output_path):
            continue
            
        mask_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        if not mask_files:
            continue
            
        # 收集所有掩码及其ID和面积
        masks_info = []
        for mask_file in mask_files:
            mask_path = os.path.join(folder_path, mask_file)
            mask_id = int(mask_file.split('.')[0])  # 文件名格式为"mask_id.png"
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # 二值化处理：任何非零值视为前景
            _, binary_mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)
            
            # 计算掩码面积（非零像素数量）
            area = np.count_nonzero(binary_mask)
            masks_info.append((mask_id, binary_mask, area))
        
        # 按面积从大到小排序
        masks_info.sort(key=lambda x: x[2], reverse=True)
        
        # 初始化合并后的掩码，背景设为-1
        merged_mask = np.full(masks_info[0][1].shape, fill_value=-1, dtype=np.int16)
        
        # 按面积从大到小应用掩码
        for mask_id, binary_mask, _ in masks_info:
            # 只覆盖当前未被标记的区域（merged_mask == -1）
            apply_mask = (binary_mask == 1) & (merged_mask == -1)
            merged_mask[apply_mask] = mask_id
        
        # 保存为NPY文件
        np.save(output_path, merged_mask)
        
        # 可视化合并后的掩码
        visualize_path = os.path.join(output_dir, 'visualization')
        os.makedirs(visualize_path, exist_ok=True)
        visualize_merged_mask(merged_mask, output_path=os.path.join(visualize_path, f"{folder_name}_vis.png"), show=False)


def split_masks(merged_mask, mask_id, resize=None):
    """
    从合并的掩码中提取指定ID的掩码，并可选地调整大小
    参数:
        merged_mask: 合并后的掩码数组 (2D numpy数组)
        mask_id: 要提取的掩码ID
        resize: 可选的调整大小参数 (元组, 形式是(width, height))
    返回:
        mask: 提取的掩码bool数组
    """
    # 从合并的掩码中提取指定ID的掩码
    mask = (merged_mask == mask_id)
    # 如果需要将调整掩码大小
    if resize is not None:
        # 将bool转换为uint8 (0和255)
        mask_uint8 = mask.astype('uint8') * 255
        # 调整大小
        mask_resized = cv2.resize(mask_uint8, resize, interpolation=cv2.INTER_NEAREST)
        # 转换回bool
        mask = mask_resized > 0
    return mask