import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt

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