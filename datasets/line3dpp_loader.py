import os
import xml.etree.ElementTree as ET
import numpy as np

def parse_line_segments(line3dpp_folder, img_id, width, height):
    """
    解析XML文件中的线段数据，并转换为指定格式的numpy数组。

    参数:
        line3dpp_folder: line3dpp重建结果的文件夹路径
        img_id: 图像ID
        width: 图像宽度
        height: 图像高度

    返回:
        segments: 包含线段信息的numpy数组，形状为(N, 4)，其中N为线段数量。
    """
    filename = 'segments_L3D++_'+ str(img_id) + '_' + str(width) + 'x' + str(height) + '.bin'
    filepath = os.path.join(line3dpp_folder, 'L3D++_data', filename)
    # 解析XML文件
    tree = ET.parse(filepath)
    root = tree.getroot()

    # 找到所有 <item> 元素
    items = root.findall(".//item")

    segments = []
    for item in items:
        x = float(item.find("x").text)
        y = float(item.find("y").text)
        z = float(item.find("z").text)
        w = float(item.find("w").text)
        segments.append([y, x, w, z])
    
    # 将列表转换为numpy数组
    segments = np.array(segments, dtype=np.float32) 
    # 端点坐标调整到范围内
    segments[:, 0] = np.clip(segments[:, 0], 0, height - 1)
    segments[:, 1] = np.clip(segments[:, 1], 0, width - 1)
    segments[:, 2] = np.clip(segments[:, 2], 0, height - 1)
    segments[:, 3] = np.clip(segments[:, 3], 0, width - 1)

    return segments  # (N, 4)