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


def parse_lines3dpp(folder_path):
    """
    解析 txt 文件，将 3D 线段存储到矩阵中，将对应的 2D 线段存储到列表中。

    参数:
    - folder_path: 包含 3D 线段数据的文件夹路径。

    返回:
    - three_d_lines: numpy 数组，存储所有 3D 线段，形状为 (total_3d_lines, 6)
    - two_d_lines: 列表，存储每个 3D 线段对应的 2D 线段信息
    """
    lines3d = []  # 用于存储所有 3D 线段
    residuals2d_for_lines3d = []    # 用于存储每个 3D 线段对应的 2D 线段

    # 在指定文件夹中查找符合条件的文件
    for filename in os.listdir(folder_path):
        if filename.startswith("Line3D++") and filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            break
    
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            
            # 解析 3D 线段
            n = int(data[0])  # 3D 线段数量
            line3d = np.array(data[1:1 + n * 6], dtype=float).reshape(n, 6)
            lines3d.append(line3d)
            
            # 解析 2D 线段
            m_start_idx = 1 + n * 6
            m = int(data[m_start_idx])  # 2D 线段数量
            residuals2d = []
            for i in range(m):
                start_idx = m_start_idx + 1 + i * 6
                cam_id = int(data[start_idx])
                seg_id = int(data[start_idx + 1])
                p1x, p1y, q1x, q1y = map(float, data[start_idx + 2:start_idx + 6])
                residuals2d.append((cam_id, seg_id, p1x, p1y, q1x, q1y))
            for j in range(n):
                if n > 1:
                    a = 0
                residuals2d_for_lines3d.append(residuals2d)
    # 将所有 3D 线段合并为一个矩阵
    lines3d = np.vstack(lines3d).reshape(-1, 6)
    return lines3d, residuals2d_for_lines3d