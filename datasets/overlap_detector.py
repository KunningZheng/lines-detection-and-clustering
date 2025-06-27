import numpy as np


def match_pair(camerasInfo, points_in_images, match_point_num=0, k_near = 99999):
    '''
    统计相片之间公共特征点的数量，获得相片之间的重叠关系
    - args
        - camerasInfo:相机内外参信息
        - points_in_images:每个相片对应的3D点ID列表
        - match_point_num:匹配点的阈值,默认0,即只要有匹配点就认为有重叠
        - k_near:返回的相片数量,默认99999,即返回所有有重叠的相片
    - return
        - overlap_images:记录邻近相片（包含自己与自己匹配的情况），认为它们有重叠，按照匹配点数从大到小排序
    - 注意:返回的overlap_images是一个列表,每个元素是一个数组
    '''
    num_cameras = len(camerasInfo)
    matches_matrix = np.zeros((num_cameras, num_cameras), dtype=int)
    for camera_ids in points_in_images:
        for i in range(len(camera_ids)):
            for j in range(i, len(camera_ids)):
                matches_matrix[camera_ids[i], camera_ids[j]] += 1 
                matches_matrix[camera_ids[j], camera_ids[i]] += 1

    overlap_images = []
    for i in range(num_cameras):
        # 获取当前相机的所有匹配信息
        matches = matches_matrix[i]
        # 获取匹配数大于阈值的相机索引和匹配数
        valid_indices = np.where(matches > match_point_num)[0]
        valid_matches = matches[valid_indices]
        # 按照匹配数从大到小排序
        sorted_indices = np.argsort(-valid_matches)
        sorted_overlap = valid_indices[sorted_indices]
        sorted_overlap = sorted_overlap[:k_near]  # 只保留前k_near个
        overlap_images.append(sorted_overlap)

    return overlap_images


def find_common_points(img1_id, img2_id, camerasInfo):
    '''
    找到两张相片之间的公共特征点
    - args
        - img1_id:第一张相片的ID
        - img2_id:第二张相片的ID
        - camerasInfo:相机内外参信息
    - return
        - common_points:公共特征点的列表,每个元素是一个元组,包含两张相片中对应的2D点坐标
    '''
    # 取交集
    common_point_ids = np.intersect1d(camerasInfo[img1_id]['points3D_ids'], camerasInfo[img2_id]['points3D_ids'])

    common_points = []
    for point_id in common_point_ids:
        pt1 = camerasInfo[img1_id]['points3D_to_xys'][point_id]
        pt2 = camerasInfo[img2_id]['points3D_to_xys'][point_id]
        common_points.append((pt1, pt2))
    return np.array(common_points)


def compute_bounding_box(points):
    '''
    计算2D点的最小外接矩形
    - args
        - points:2D点的列表,每个元素是一个元组,包含点的x和y坐标
    - return
        - area:最小外接矩形的面积
    '''
    if len(points) == 0:
        return 0
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    return area