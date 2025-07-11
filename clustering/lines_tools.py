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
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
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