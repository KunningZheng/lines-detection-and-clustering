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


# internal
from datasets.sfm_reader import load_sparse_model
from datasets.overlap_detector import match_pair
from datasets.mask_processor import merge_masks_to_npy

if __name__ == '__main__':
    ####################################### 需要手动改变的参数 #######################################
    # 设置Workspace路径
    workspace_path = '/home/rylynn/Pictures/Clustering_Workspace'
    # 设置场景名称
    scene_name = 'group_selectByPt_chunk3'
    # 取前k_near个相片为邻近视角
    k_near = 2

    ####################################### 预处理 #######################################
    sparse_model_path = os.path.join(workspace_path, scene_name, 'Colmap','sparse')
    images_path = os.path.join(workspace_path, scene_name, 'Colmap', 'images')
    line3dpp_path = os.path.join(workspace_path, scene_name, 'Line3D++')
    single_mask_path = os.path.join(workspace_path, scene_name, 'SAM_Mask', 'Single_Mask')
    merged_mask_path = os.path.join(workspace_path, scene_name, 'SAM_Mask', 'Merged_Mask')

    # 预处理1：读取sparse model
    camerasInfo, points_in_images = load_sparse_model(sparse_model_path)
    # 预处理2：统计相片之间公共特征点的数量
    match_matrix = match_pair(camerasInfo, points_in_images, k_near=k_near)
    # 预处理3：形成merged mask
    merge_masks_to_npy(single_mask_path, merged_mask_path)


    img_info = {}
    ####################### 每张航片循环 #######################
    for img_id, img_filename in img_info.items():
        ### 查找临近航片 ###
        

        ####################### 每个mask循环 #######################
        for mask_id in range(1, 6):  # 假设有5个mask
            # 读取当前mask中的线段

            ####################### 每张临近航片循环 #######################
            for neighbor_id in range(1, 6): # 假设有5个临近航片
                testpp = 0
                ### step1:2D线段对应的3D线段投影到临近航片 ###
                # 临近行片的栅格图
                
                ### step2:线段可见性判断 ###

                ### step3:基于可见线段对应的mask投票 ###
                # 共视线段

                ### step4:统计有效投票 ###

                ### step5:判断掩码有效性并记录有效关联 ###

### 建立跨视角的掩码关联 ###

### 统计线段聚类实例 ###