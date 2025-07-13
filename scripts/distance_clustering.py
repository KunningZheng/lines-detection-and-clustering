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
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt
import open3d as o3d


# internal
from datasets.sfm_reader import load_sparse_model
from datasets.overlap_detector import match_pair
from datasets.mask_processor import merge_masks_to_npy
from datasets.line3dpp_loader import parse_line_segments, parse_lines3dpp
from clustering.mask_association import load_mask_lines_association, associate_projline_to_mask
from clustering.lines_correspondence import LineCorrespondence
from clustering.lines_tools import visualize_line_clusters, determine_line3d_visibility
from clustering.clustering_methods import Chinese_Whispers, Leiden_community, Node2Vec_HDBSCAN, similarity_HDBCAN, Leiden_community_normalized




def find_ground_plane(lines, height_threshold=0.1):
    """
    找到地面平面（假设地面是z值最低的平面）
    
    参数:
        lines: 三维线段数组，形状(N,6)，每行是(x1,y1,z1,x2,y2,z2)
        height_threshold: 被认为是地面的高度阈值
    
    返回:
        ground_z: 地面的z坐标
    """
    # 计算所有点的z坐标
    all_z = np.concatenate([lines[:, 2], lines[:, 5]])
    
    # 找到最低的z值作为地面
    ground_z = np.min(all_z) + height_threshold
    
    return ground_z

def project_lines_to_ground(lines, ground_z):
    """
    将线段投影到地面平面
    
    参数:
        lines: 三维线段数组，形状(N,6)
        ground_z: 地面的z坐标
    
    返回:
        projected_lines: 投影后的线段，形状(N,4)，每行是(x1,y1,x2,y2)
    """
    projected_lines = np.zeros((lines.shape[0], 4))
    
    for i in range(lines.shape[0]):
        x1, y1, z1, x2, y2, z2 = lines[i]
        
        projected_lines[i] = [x1, y1, x2, y2]

    
    return projected_lines


def calculate_projection_distance(line1, line2):
    """
    计算两条线段之间的投影距离
    
    参数:
        line1: 第一条线段 (x1,y1,x2,y2)
        line2: 第二条线段 (x1,y1,x2,y2)
    
    返回:
        最小投影距离
    """
    def point_to_line_distance(point, line):
        """计算点到线段的距离"""
        x, y = point
        x1, y1, x2, y2 = line
        
        # 线段长度
        line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if line_length == 0:
            return np.sqrt((x-x1)**2 + (y-y1)**2)
        
        # 计算投影比例
        t = ((x-x1)*(x2-x1) + (y-y1)*(y2-y1)) / (line_length**2)
        
        # 限制t在[0,1]范围内
        t = max(0, min(1, t))
        
        # 投影点坐标
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # 计算距离
        return np.sqrt((x-proj_x)**2 + (y-proj_y)**2)
    
    # 提取线段端点
    p1 = line1[:2]
    p2 = line1[2:]
    p3 = line2[:2]
    p4 = line2[2:]
    
    # 计算所有端点之间的投影距离
    d1 = point_to_line_distance(p1, line2)
    d2 = point_to_line_distance(p2, line2)
    d3 = point_to_line_distance(p3, line1)
    d4 = point_to_line_distance(p4, line1)
    
    # 返回最小的距离
    return min(d1, d2, d3, d4)


def cluster_lines_hdbscan_projection(projected_lines, min_cluster_size=5, min_samples=None):
    """
    使用HDBSCAN和投影距离对线段进行聚类
    
    参数:
        projected_lines: 投影后的线段，形状(N,4)
        min_cluster_size: 最小簇大小
        min_samples: HDBSCAN参数
    
    返回:
        labels: 每个线段的簇标签
    """
    # 计算距离矩阵
    n_lines = projected_lines.shape[0]
    distance_matrix = np.zeros((n_lines, n_lines))
    
    for i in tqdm(range(n_lines), desc="Calculating distance matrix"):
        for j in range(i+1, n_lines):
            dist = calculate_projection_distance(projected_lines[i], projected_lines[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    # 运行HDBSCAN聚类
    if min_samples is None:
        min_samples = min_cluster_size
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                               min_samples=min_samples,
                               metric='precomputed')
    labels = clusterer.fit_predict(distance_matrix)
    
    return labels

def visualize_projected_lines(projected_lines, labels):
    """
    使用matplotlib可视化投影线段的聚类结果
    
    参数:
        projected_lines: 投影后的线段，形状(N,4)
        labels: 聚类标签
    """
    plt.figure(figsize=(10, 8))
    
    # 获取唯一的标签值
    unique_labels = np.unique(labels)
    
    # 为每个簇分配颜色
    # 为每个簇生成随机RGB颜色
    np.random.seed(42)  # 固定随机种子保证可重复性
    colors = np.random.rand(len(unique_labels), 3)
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            # 噪声点用灰色表示
            color = 'gray'
            alpha = 0.3
        else:
            color = colors[i]
            alpha = 0.7
        
        # 获取当前标签的所有线段
        mask = labels == label
        lines = projected_lines[mask]
        
        # 绘制线段
        for line in lines:
            x1, y1, x2, y2 = line
            plt.plot([x1, x2], [y1, y2], color=color, alpha=alpha, 
                    label=f'Cluster {label}' if i == 0 else "")
    
    plt.title('Projected 2D Lines with HDBSCAN Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    
    
    plt.show()


def process_lines(lines, intermediate_output_path, min_cluster_size=5, min_samples=None):
    """
    改进后的处理线段的完整流程
    
    参数:
        lines: 三维线段数组，形状(N,6)
        min_cluster_size: 最小簇大小
        min_samples: HDBSCAN参数
    
    返回:
        projected_lines: 投影后的线段
        labels: 聚类标签
    """
    # 1. 找到地面平面
    ground_z = find_ground_plane(lines)
    
    # 2. 将线段投影到地面
    projected_lines = project_lines_to_ground(lines, ground_z)
    
    # 3. 使用改进的HDBSCAN聚类（基于投影距离）
    #labels = cluster_lines_hdbscan_projection(projected_lines, min_cluster_size, min_samples)
    # 保存聚类标签到文件
    labels_output_path = os.path.join(intermediate_output_path, 'cluster_labels.npy')
    #np.save(labels_output_path, labels)
    #print(f"Cluster labels saved to {labels_output_path}")
    # 4. 可视化结果
    #visualize_projected_lines(projected_lines, labels)  # 2D投影可视化

    labels = np.load(labels_output_path)
    # 4. 创建聚类结果字典
    line3d_cluster = {}
    
    # 获取所有非噪声的cluster ID（排除-1）
    unique_clusters = set(labels) - {-1}
    
    for cluster_id in unique_clusters:
        # 获取属于当前cluster的所有line3d的索引
        line_indices = np.where(labels == cluster_id)[0].tolist()
        line3d_cluster[cluster_id] = line_indices
    
    visualize_line_clusters(lines, line3d_cluster)
    return projected_lines, labels


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

    # 预处理5：建立同名线段的关联
    # 解析数据文件
    lines3d, residuals2d_for_lines3d = parse_lines3dpp(line3dpp_path)

    x_range = max(lines3d[:,0]) - min(lines3d[:,0])
    y_range = max(lines3d[:,1]) - min(lines3d[:,1])
    z_range = max(lines3d[:,2]) - min(lines3d[:,2])
    
    # 处理数据
    projected_lines, labels = process_lines(lines3d, intermediate_output_path, min_cluster_size=10)
    
    # 打印聚类结果
    print(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"Points labeled as noise: {np.sum(labels == -1)}")