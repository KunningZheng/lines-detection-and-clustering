import os
import cv2
import json
import numpy as np
from clustering.lines_tools import visualize_line_clusters


def lines3d_gt_generator(lines3d, residuals2d_for_lines3d, lines2d_instanceID_gt, output_path, show=False):
    if os.path.exists(os.path.join(output_path, 'lines3d_clusters_gt.json')):
        print('loading existing lines3d_clusters_gt.json file')
        with open(os.path.join(output_path, 'lines3d_clusters_gt.json'), 'r') as f:
            lines3d_clusters_gt = json.load(f)
        # Convert keys to int
        lines3d_clusters_gt = {int(instanceID): line_ids for instanceID, line_ids in lines3d_clusters_gt.items()}
    else:
        print('generating lines3d clusters groundtruth')
        #----------- 1.循环3D线段，得到3D线段的实例ID -----------#
        lines3d_instanceID_gt = -1 * np.ones((lines3d.shape[0], 1))
        i = 0
        for correspond_lines2d in residuals2d_for_lines3d:
            correspond_lines2d_instanceID = []
            for line2d in correspond_lines2d:
                cam_id, line2d_id = line2d[0], line2d[1]
                # 查找line2d是否有对应的instanceID
                line2d_instanceID = lines2d_instanceID_gt[cam_id].get(line2d_id, None)
                if line2d_instanceID is not None:
                    correspond_lines2d_instanceID.append(line2d_instanceID)
            # 如果correspond_lines2d_instanceID不为空
            if len(correspond_lines2d_instanceID) != 0:
                # 取出现频率最高的instanceID
                line3d_instanceID = max(set(correspond_lines2d_instanceID), key=correspond_lines2d_instanceID.count)
                lines3d_instanceID_gt[i] = int(line3d_instanceID)
            i = i + 1

        #----------- 2.lines3d_instanceID转成字典形式 -----------#
        unique_instanceIDs = np.unique(lines3d_instanceID_gt)
        # 以instanceID为键，所有instanceID相同的lines3d的ID作为值
        lines3d_clusters_gt = {int(instanceID): [] for instanceID in unique_instanceIDs if instanceID != -1}
        for line3d_id, instanceID in enumerate(lines3d_instanceID_gt):
            instanceID = int(instanceID)
            if instanceID != -1:
                lines3d_clusters_gt[instanceID].append(line3d_id)
        # 保存成json文件
        with open(os.path.join(output_path, 'lines3d_clusters_gt.json'), 'w') as f:
            json.dump(lines3d_clusters_gt, f)

    #----------- 3.可视化实例ID赋色的3Dline -----------#
    if show:
        visualize_line_clusters(lines3d, lines3d_clusters_gt)
    
    return lines3d_clusters_gt
