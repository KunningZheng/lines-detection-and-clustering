import numpy as np
from collections import defaultdict

class LineCorrespondence:
    def __init__(self, lines2d):
        """
        初始化线段对应关系数据结构
        
        参数:
        - lines2d: parse_lines3dpp 返回的 2D 线段列表
        """
        # 数据结构1: 从 (相机ID, 线段ID) 映射到 3D 线段索引
        self.cam_seg_to_line3d = {}
        
        # 数据结构2: 从 3D 线段索引映射到 {相机ID: 线段ID}
        self.line3d_to_cam_seg = defaultdict(dict)
        
        # 构建数据结构
        for line3d_idx, projections in enumerate(lines2d):
            for proj in projections:
                cam_id = proj[0]
                seg_id = proj[1]
                
                # 将 (相机ID, 线段ID) 映射到 3D 线段索引
                self.cam_seg_to_line3d[(cam_id, seg_id)] = line3d_idx
                
                # 将 3D 线段索引映射到相机ID和线段ID
                self.line3d_to_cam_seg[line3d_idx][cam_id] = seg_id
    
    def find_correspondence(self, src_cam, src_seg_id, tgt_cam):
        """
        查询源图像中的线段在目标图像中的对应线段
        
        参数:
        - src_cam: 源图像相机ID
        - src_seg_id: 源图像中的线段ID
        - tgt_cam: 目标图像相机ID
        
        返回:
        - 目标图像中的线段ID (如果存在对应线段)
        - None (如果没有对应线段)
        """
        # 步骤1: 找到源线段对应的 3D 线段
        key = (src_cam, src_seg_id)
        if key not in self.cam_seg_to_line3d:
            return None  # 没有找到对应的 3D 线段
        
        line3d_idx = self.cam_seg_to_line3d[key]
        
        # 步骤2: 找到该 3D 线段在目标相机中的投影
        cam_seg_dict = self.line3d_to_cam_seg[line3d_idx]
        return cam_seg_dict.get(tgt_cam, None)
    
    
    def find_line3d_by_cam_seg(self, cam_id, seg_id):
        """
        根据相机ID和线段ID查找对应的3D线段索引
        
        参数:
        - cam_id: 相机ID
        - seg_id: 线段ID
        
        返回:
        - 对应的3D线段索引 (如果存在)
        - None (如果没有对应的3D线段)
        """
        return self.cam_seg_to_line3d.get((cam_id, seg_id), None)