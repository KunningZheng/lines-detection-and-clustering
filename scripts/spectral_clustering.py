# FOR DEBUGGING ONLY
import sys
from pathlib import Path

# 获取当前脚本的父目录的父目录（即 your_project/）
project_root = Path(__file__).parent.parent
# 添加到 Python 路径
sys.path.append(str(project_root))

# external
import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import deque
from sklearn.cluster import SpectralClustering

# internal
from datasets.sfm_reader import load_sparse_model
from datasets.overlap_detector import match_pair
from datasets.mask_processor import merge_masks_to_npy
from datasets.line3dpp_loader import parse_line_segments, parse_lines3dpp
from clustering.mask_association import load_mask_lines_association
from clustering.lines_correspondence import LineCorrespondence
from clustering.lines_tools import visualize_line_clusters


def cluster_3d_segments(mask_assignments, required_views=3, n_clusters=None):
    """
    使用图割算法(谱聚类)对3D线段进行聚类
    
    参数:
        mask_assignments: 一个字典，键是视角名称，值是字典{线段索引: mask_id}
        required_views: 需要多少个视角下mask相同才认为属于同一类
        n_clusters: 聚类数量，如果为None则自动确定
        
    返回:
        一个列表，包含所有聚类结果，每个聚类是一个线段索引的集合
    """
    if not mask_assignments:
        return []
    
    # 收集所有线段索引
    all_segments = set()
    for view_data in mask_assignments.values():
        all_segments.update(view_data.keys())
    all_segments = sorted(all_segments)
    
    # 创建线段索引到连续整数的映射
    segment_to_idx = {seg: idx for idx, seg in enumerate(all_segments)}
    idx_to_segment = {idx: seg for seg, idx in segment_to_idx.items()}
    num_segments = len(segment_to_idx)
    
    # 如果没有线段，返回空结果
    if num_segments == 0:
        return []
    
    # 构建相似度矩阵
    similarity_matrix = np.zeros((num_segments, num_segments))
    
    from itertools import combinations
    from collections import defaultdict
    
    # 创建线段对到共享视角数的映射
    pair_shared_views = defaultdict(int)
    
    # 对于每个视角，记录哪些mask对应哪些线段
    for view_data in mask_assignments.values():
        # 创建mask到线段列表的映射
        mask_to_segments = defaultdict(list)
        for seg, mask in view_data.items():
            mask_to_segments[mask].append(seg)
        
        # 对每个mask下的所有线段对增加共享计数
        for seg_list in mask_to_segments.values():
            if len(seg_list) > 1:
                for seg1, seg2 in combinations(seg_list, 2):
                    if seg1 < seg2:
                        pair = (seg1, seg2)
                    else:
                        pair = (seg2, seg1)
                    pair_shared_views[pair] += 1
    
    # 填充相似度矩阵
    for pair, count in pair_shared_views.items():
        if count >= required_views:
            seg1, seg2 = pair
            i = segment_to_idx[seg1]
            j = segment_to_idx[seg2]
            similarity_matrix[i, j] = count
            similarity_matrix[j, i] = count
    
    # 如果没有足够的相似性信息，每个线段自成一类
    if np.sum(similarity_matrix) == 0:
        return [{seg} for seg in all_segments]
    
    # 自动确定聚类数量(基于矩阵的秩或特征值)
    if n_clusters is None:
        # 简单启发式方法：使用非零特征值的数量
        eigenvalues = np.linalg.eigvalsh(similarity_matrix)
        n_clusters = np.sum(eigenvalues > 1e-3)
        n_clusters = max(1, min(n_clusters, num_segments))
    
    # 使用谱聚类
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='discretize',
        random_state=42
    ).fit(similarity_matrix)
    
    # 收集聚类结果
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        seg = idx_to_segment[idx]
        clusters.setdefault(label, set()).add(seg)
    
    return clusters


if __name__ == '__main__':
    ####################################### 需要手动改变的参数 #######################################
    # 设置Workspace路径
    workspace_path = '/home/rylynn/Pictures/Clustering_Workspace'
    # 设置场景名称
    scene_name = 'group_selectByPt_chunk3'
    # 取前k_near个相片为邻近视角
    k_near = 10

    ####################################### 预处理 #######################################
    sparse_model_path = os.path.join(workspace_path, scene_name, 'Colmap','sparse')
    images_path = os.path.join(workspace_path, scene_name, 'Colmap', 'images')
    line3dpp_path = os.path.join(workspace_path, scene_name, 'Line3D++')
    single_mask_path = os.path.join(workspace_path, scene_name, 'SAM_Mask', 'Single_Mask')
    merged_mask_path = os.path.join(workspace_path, scene_name, 'SAM_Mask', 'Merged_Mask')

    # 创建输出目录
    intermediate_output_path = os.path.join(workspace_path, scene_name, 'intermediate_outputs')
    os.makedirs(intermediate_output_path, exist_ok=True)

    # 预处理1：读取sparse model
    camerasInfo, points_in_images = load_sparse_model(sparse_model_path)
    # 预处理2：统计相片之间公共特征点的数量
    match_matrix = match_pair(camerasInfo, points_in_images, k_near=k_near)
    # 预处理3：形成merged mask
    merge_masks_to_npy(single_mask_path, merged_mask_path)
    # 预处理4：关联mask和线段
    all_mask_to_lines, all_line_to_mask = load_mask_lines_association(
        camerasInfo, 
        merged_mask_path, 
        line3dpp_path, 
        intermediate_output_path
    )
    # 预处理5：建立同名线段的关联
    # 解析数据文件
    lines3d, residuals2d_for_lines3d = parse_lines3dpp(line3dpp_path)
    # 初始化对应关系结构
    line_corr = LineCorrespondence(residuals2d_for_lines3d)

    # 初始化跨视角的掩码关联
    all_line3d_to_mask = {}
    
    for cam_id, line_to_mask in all_line_to_mask.items():
        cam_id = int(cam_id)
        line3d_to_mask = {}
        for seg_id, mask_id in line_to_mask.items():
            line3d_id = line_corr.find_line3d_by_cam_seg(cam_id, int(seg_id))
            if line3d_id is not None:
                line3d_to_mask[line3d_id] = mask_id
        all_line3d_to_mask[cam_id] = line3d_to_mask
        
    lines3d_clusters = cluster_3d_segments(all_line3d_to_mask, required_views=1)
    visualize_line_clusters(lines3d, lines3d_clusters)