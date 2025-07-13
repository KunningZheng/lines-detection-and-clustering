import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
import open3d as o3d


def rasterize_lines(image_shape, lines, show=False):
    """
    将线段栅格化，生成一个与图片同尺寸的 numpy 数组，每个像素存储线段编号。
    
    参数：
    - image_shape: (H, W) 代表输出栅格的高度和宽度
    - lines: 线段列表，每条线段的格式为 (x1, y1, x2, y2)
    - show: 是否显示中间结果（可选，默认为 False）
    
    返回：
    - raster: 2D numpy 数组，与 image_shape 相同，包含线段编号，未被线段覆盖的像素为 -1
    """
    H, W = image_shape
    raster_lines = np.full((H, W), -1, dtype=int)  # 初始化栅格，未被覆盖的像素设为 -1
    
    for idx, (y1, x1, y2, x2) in enumerate(lines):
        # 计算线段的像素点
        rr, cc = line(round(y1), round(x1), round(y2), round(x2))  # skimage.draw.line 返回行列索引（y, x）
        
        # 过滤掉超出范围的点
        valid_idx = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = rr[valid_idx], cc[valid_idx]
        
        # 在栅格上标记线段编号
        raster_lines[rr, cc] = idx

    # 可视化
    if show:
        raster = np.where(raster_lines == -1, 255, 0)
        plt.figure(figsize=(10, 10), dpi=100)
        plt.imshow(raster, cmap='gray')

    return raster_lines


# 可视化聚类结果
def visualize_line_clusters(lines3d, clusters):
    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Line Clusters', width=1200, height=800)
    
    # 为每个聚类生成不同颜色
    n_clusters = len(clusters)
    colors = np.random.rand(n_clusters, 3)
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
    vis.add_geometry(coordinate_frame)
    
    # 遍历每个聚类
    for i, (cluster_id, line_data) in enumerate(clusters.items()):
        cluster_color = colors[i][:3]  # 取RGB值
        
        # 创建当前聚类的线段集合
        line_set = o3d.geometry.LineSet()
        points = []
        lines = []
        
        # 添加当前聚类的所有线段
        for idx, line_id in enumerate(line_data):
            x1, y1, z1, x2, y2, z2 = lines3d[line_id]
            points.append([x1, y1, z1])
            points.append([x2, y2, z2])
            lines.append([2*idx, 2*idx+1])  # 连接两个点形成线段
        
        line_set.points = o3d.utility.Vector3dVector(np.array(points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
        line_set.paint_uniform_color(cluster_color)
        
        vis.add_geometry(line_set)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.line_width = 5.0  # 设置线段粗细
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深色背景
    
    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


def determine_line3d_visibility(cam_dict, line3d, points3d_xyz):

    # 获取当前相机参数
    cam_id = cam_dict['id']
    R = cam_dict['rotation']
    T = cam_dict['position']
    fx = cam_dict['fx']
    fy = cam_dict['fy']
    width = cam_dict['width']
    height = cam_dict['height']
    
    # 获取3D线段的两个端点
    p1 = line3d[:3]
    p2 = line3d[3:]
    
    # 计算线段中心点
    center = (p1 + p2) / 2
    
    # ===  1.线段中心点与相机的夹角（视线方向与线段方向的夹角） ===
    view_dir = center - T
    view_dir = view_dir / np.linalg.norm(view_dir)
    line_dir = p2 - p1
    line_dir = line_dir / np.linalg.norm(line_dir)
    angle = np.arccos(np.dot(view_dir, line_dir))
    
    # 如果夹角接近90度（垂直），线段在视角下几乎不可见
    if abs(angle - np.pi/2) < np.pi/6:  # 30度阈值
        return False


    # ===  2.线段端点投影是否落入图像范围  ===
    # 将3D点转换到相机坐标系
    p1_cam = R @ p1 + T
    p2_cam = R @ p2 + T
    
    # 检查深度是否为正（在相机前方）
    if p1_cam[2] <= 0 or p2_cam[2] <= 0:
        return False
        
    # 投影到图像平面
    p1_proj = np.array([fx * p1_cam[0]/p1_cam[2], fy * p1_cam[1]/p1_cam[2]])
    p2_proj = np.array([fx * p2_cam[0]/p2_cam[2], fy * p2_cam[1]/p2_cam[2]])
    
    # 检查投影点是否在图像范围内
    if (p1_proj[0] < 0 or p1_proj[0] >= width or 
        p1_proj[1] < 0 or p1_proj[1] >= height or
        p2_proj[0] < 0 or p2_proj[0] >= width or 
        p2_proj[1] < 0 or p2_proj[1] >= height):
        return False
        
    # ===  3.利用稀疏点云遮挡测试  ===
    # 计算线段在图像上的投影长度
    #proj_length = np.linalg.norm(p1_proj - p2_proj)
    #if proj_length < 10:  # 投影太短的线段不考虑
        #return False
        
    # 检查线段中间是否有稀疏点云遮挡
    # 采样线段上的点进行检查
    for alpha in np.linspace(0.1, 0.9, 5):  # 采样5个点
        sample_point = p1 * alpha + p2 * (1 - alpha)
        sample_cam = R @ sample_point + T
        
        # 投影到图像平面
        sample_proj = np.array([fx * sample_cam[0]/sample_cam[2], fy * sample_cam[1]/sample_cam[2]])
        
        # 检查该点附近是否有更近的3D点
        for pt_id in cam_dict['points3D_ids']:
            pt = points3d_xyz[pt_id]
            pt_cam = R @ pt + T
            if pt_cam[2] <= 0:  # 点在相机后方
                continue
                
            pt_proj = np.array([fx * pt_cam[0]/pt_cam[2], fy * pt_cam[1]/pt_cam[2]])
            dist = np.linalg.norm(pt_proj - sample_proj)
            
            # 如果附近有点且深度更小（更近），则认为被遮挡
            if dist < 10 and pt_cam[2] < sample_cam[2] * 0.9:
                return False
    # 行号在前，列号在后
    return (p1_proj[::-1], p2_proj[::-1])