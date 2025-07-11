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

# internal
from datasets.sfm_reader import load_sparse_model
from datasets.overlap_detector import match_pair
from datasets.mask_processor import merge_masks_to_npy
from datasets.line3dpp_loader import parse_line_segments, parse_lines3dpp
from clustering.mask_association import load_mask_lines_association
from clustering.lines_correspondence import LineCorrespondence
from clustering.lines_tools import visualize_line_clusters

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
    all_masks_association = {}
    
    ####################### 每张航片循环 #######################
    for cam_dict in tqdm(camerasInfo, desc="Computing mask to mask associations"):
        img_name = cam_dict['img_name'].split('/')[-1]
        width, height = int(cam_dict['width']), int(cam_dict['height'])

        # 读取当前航片对应的2D线段
        lines2d = parse_line_segments(line3dpp_path, cam_dict['id'], width, height)
        # 读取当前航片mask和2D线段的关联
        mask_to_lines = all_mask_to_lines.get(str(cam_dict['id']), {})
        # 查找临近航片
        neighbor_ids = match_matrix[cam_dict['id']][1:]
        
        ####################### 每个mask循环 #######################
        for mask_id,associated_lines_id in mask_to_lines.items():
            mask_id = int(mask_id)  # 确保mask_id是整数类型
            # 读取当前mask中的线段
            associated_lines2d = lines2d[associated_lines_id]
            # 初始化投票向量
            neighbor_votes=[]
            neighbor_masks=[]
            ####################### 每张临近航片循环 #######################
            for neighbor_id in neighbor_ids:
                neighbor_id = int(neighbor_id)  # 确保邻近航片ID是整数类型
                ### step1:查找同名线段，并记录其对应的mask_id ###
                corr_seg_masks = []
                for src_seg_id in associated_lines_id:
                    corresponding_seg = line_corr.find_correspondence(
                        cam_dict['id'], 
                        src_seg_id, 
                        neighbor_id)
                    if corresponding_seg is not None:
                        corr_mask_id = all_line_to_mask[str(neighbor_id)].get(str(corresponding_seg), None)
                        if corr_mask_id is not None:
                            # 如果找到了对应的mask，则记录下来
                            corr_seg_masks.append(corr_mask_id)
                ### step2:根据同名线段的mask数量投票 ###
                if corr_seg_masks != []:
                    unique_masks = set(corr_seg_masks)
                    vote = len(unique_masks)
                    if vote == 1:
                        neighbor_masks.append(corr_seg_masks[0])  # 记录唯一的mask
                    else:
                        # 如果有多个mask，则记录所有的mask
                        neighbor_masks.extend(unique_masks)
                    neighbor_votes.append(vote)
                    # vote=0，能见部分少
                    # vote=1，在邻近航片中有唯一mask
                    # vote>1，在邻近航片中有多个mask
                else:
                    neighbor_votes.append(0)  # 没有找到对应的mask
                    neighbor_masks.append(-1)  # -1表示没有对应的mask

            ### step3：判定当前mask是否是correct mask ###
            # 如果是false mask，则跳过
            if max(neighbor_votes) != 1:
                continue
            # 如果是correct mask
            else:
                ####################### 每张临近航片循环 #######################
                for idx in range(len(neighbor_ids)):
                    # 如果vote=1，则为有效投票，建立跨视角的掩码关联
                    if neighbor_votes[idx] == 1:
                        neighbor_id = int(neighbor_ids[idx])
                        # 记录当前mask和邻近航片的关联
                        all_masks_association.setdefault((cam_dict['id'], mask_id), []).append((neighbor_id, neighbor_masks[idx]))
                        all_masks_association.setdefault((neighbor_id, neighbor_masks[idx]), []).append((cam_dict['id'], mask_id))       


### 统计视角间关联的实例 ###


# 创建节点集合
all_nodes = set(all_masks_association.keys())
visited = set()
masks_clusters = {}
cluster_id = 0

# 使用BFS遍历图查找连通分量
for node in all_nodes:
    if node not in visited:
        # 初始化新簇
        current_cluster = []
        queue = deque([node])
        visited.add(node)
        
        while queue:
            current_node = queue.popleft()
            current_cluster.append(current_node)
            
            # 遍历所有邻居
            for neighbor in all_masks_association.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # 存储当前簇
        masks_clusters[cluster_id] = current_cluster
        cluster_id += 1

# 打印聚类结果
print(f"Found {len(masks_clusters)} instance clusters")
#for cluster_id, masks in masks_clusters.items():
    #print(f"Cluster {cluster_id} contains {len(masks)} masks: {masks}")

# 聚类3D线段
lines3d_clusters = {}
for cluster_id, masks in masks_clusters.items():
    for mask in masks:
        cam_id, mask_id = mask
        # 获取当前mask对应的2D线段
        associated_lines = all_mask_to_lines.get(str(cam_id), {}).get(str(mask_id), [])
        # 获取2D线段对应的3D线段
        for seg_id in associated_lines:
            line3d_id = line_corr.find_line3d_by_cam_seg(cam_id, seg_id)
            if line3d_id is not None:
                lines3d_clusters.setdefault(cluster_id, set()).add(line3d_id)
# 选择前100个线段最多的cluster
largest_clusters = sorted(lines3d_clusters.values(), key=lambda cluster: len(cluster), reverse=True)[:10]
# 将largest_clusters转换为字典
largest_clusters = {i: cluster for i, cluster in enumerate(largest_clusters)}
# 可视化3D线段聚类结果
visualize_line_clusters(lines3d, largest_clusters)