# FOR DEBUGGING ONLY
import sys
from pathlib import Path

# 获取当前脚本的父目录的父目录（即 your_project/）
project_root = Path(__file__).parent.parent
# 添加到 Python 路径
sys.path.append(str(project_root))

import os
import numpy as np
from collections import defaultdict
from itertools import combinations
from node2vec import Node2Vec
import networkx as nx
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler

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
    使用 Node2Vec + HDBSCAN 对 3D 线段进行聚类
    
    参数:
        mask_assignments: 字典 {视角: {线段索引: mask_id}}
        required_views: 最少共现视角数
        n_clusters: 保留参数（HDBSCAN自动确定）
        
    返回:
        聚类结果列表，每个聚类是线段索引的集合
    """
    # === 1. 数据准备 ===
    all_segments = set()
    for view_data in mask_assignments.values():
        all_segments.update(view_data.keys())
    all_segments = sorted(all_segments)
    
    segment_to_idx = {seg: idx for idx, seg in enumerate(all_segments)}
    idx_to_segment = {idx: seg for seg, idx in segment_to_idx.items()}
    
    if not segment_to_idx:
        return []

    # === 2. 构建加权图 ===
    G = nx.Graph()
    for seg in all_segments:
        G.add_node(segment_to_idx[seg])
    
    # 计算共现权重
    co_occurrence = defaultdict(int)
    for view_data in mask_assignments.values():
        mask_to_segments = defaultdict(list)
        for seg, mask in view_data.items():
            mask_to_segments[mask].append(seg)
        
        for seg_list in mask_to_segments.values():
            for seg1, seg2 in combinations(seg_list, 2):
                if seg1 < seg2:
                    pair = (seg1, seg2)
                else:
                    pair = (seg2, seg1)
                co_occurrence[pair] += 1
    
    # 添加边（仅保留满足阈值的连接）
    for (seg1, seg2), count in co_occurrence.items():
        if count >= required_views:
            G.add_edge(segment_to_idx[seg1], segment_to_idx[seg2], weight=count)
    
    if not G.edges():
        return [{seg} for seg in all_segments]

    # === 3. Node2Vec 嵌入 ===
    node2vec = Node2Vec(
        G,
        dimensions=64,       # 嵌入维度
        walk_length=30,       # 随机游走长度
        num_walks=200,        # 每个节点的游走次数
        weight_key='weight',  # 使用边权重
        workers=4            # 并行线程
    )
    
    model = node2vec.fit(window=10, min_count=1)
    embeddings = np.array([model.wv[str(i)] for i in range(len(all_segments))])

    # === 4. HDBSCAN 聚类 ===
    # 标准化嵌入向量
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    
    clusterer = HDBSCAN(
        min_cluster_size=5,          # 最小簇大小
        min_samples=3,               # 核心点所需邻居数
        cluster_selection_method='eom',  # 选择聚类方法
        metric='euclidean'           # 距离度量
    )
    clusterer.fit(X_scaled)
    
    # === 5. 整理结果 ===
    clusters = defaultdict(set)
    for idx, label in enumerate(clusterer.labels_):
        if label != -1:  # 过滤噪声点（HDBSCAN将噪声标记为-1）
            seg = idx_to_segment[idx]
            clusters[label].add(seg)
    
    # 将噪声点各自作为单独聚类
    noise_indices = np.where(clusterer.labels_ == -1)[0]
    for noise_idx in noise_indices:
        seg = idx_to_segment[noise_idx]
        clusters[len(clusters)] = {seg}

    # 可视化嵌入空间（2D投影）
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2)
    X_2d = tsne.fit_transform(embeddings)

    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusterer.labels_, cmap='Spectral', s=5)
    plt.colorbar()
    plt.show()
    
    return list(clusters.values())

if __name__ == '__main__':
    ####################################### 需要手动改变的参数 #######################################
    # 设置Workspace路径
    workspace_path = '/home/rylynn/Pictures/Clustering_Workspace'
    # 设置场景名称
    scene_name = 'group_selectByPt_chunk3'  # group_selectByPt_chunk3
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
    # 将largest_clusters转换为字典
    lines3d_clusters = {i: cluster for i, cluster in enumerate(lines3d_clusters)}
    #largest_clusters = sorted(lines3d_clusters.values(), key=lambda cluster: len(cluster), reverse=True)[:10]
    # 将largest_clusters转换为字典    
    #largest_clusters = {i: cluster for i, cluster in enumerate(largest_clusters)}
    visualize_line_clusters(lines3d, lines3d_clusters)