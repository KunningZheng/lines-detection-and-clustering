import numpy as np
from tqdm import tqdm
from itertools import combinations
from node2vec import Node2Vec
import networkx as nx
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from collections import defaultdict

def Chinese_Whispers(mask_assignments, required_views=3, n_clusters=None):
    """
    使用 Chinese Whispers (CW) 算法对 3D 线段进行聚类
    
    参数:
        mask_assignments: 一个字典，键是视角名称，值是字典{线段索引: mask_id}
        required_views: 需要多少个视角下 mask 相同才认为属于同一类
        n_clusters: 聚类数量（CW 自动确定，此参数仅保留接口兼容）
        
    返回:
        一个列表，包含所有聚类结果，每个聚类是一个线段索引的集合
    """
    if not mask_assignments:
        return []
    
    # 收集所有线段索引
    all_segments = set()
    for view_data in mask_assignments.values():
        all_segments.update(view_data.keys())
    all_segments = sorted(all_segments)
    
    # 创建线段索引到连续整数的映射
    segment_to_idx = {seg: idx for idx, seg in enumerate(all_segments)}
    idx_to_segment = {idx: seg for seg, idx in segment_to_idx.items()}
    num_segments = len(segment_to_idx)
    
    # 如果没有线段，返回空结果
    if num_segments == 0:
        return []
    
    # 构建相似度矩阵
    similarity_matrix = np.zeros((num_segments, num_segments))
    
    from itertools import combinations
    from collections import defaultdict
    
    # 创建线段对到共享视角数的映射
    pair_shared_views = defaultdict(int)
    
    # 对于每个视角，记录哪些 mask 对应哪些线段
    for view_data in mask_assignments.values():
        # 创建 mask 到线段列表的映射
        mask_to_segments = defaultdict(list)
        for seg, mask in view_data.items():
            mask_to_segments[mask].append(seg)
        
        # 对每个 mask 下的所有线段对增加共享计数
        for seg_list in mask_to_segments.values():
            if len(seg_list) > 1:
                for seg1, seg2 in combinations(seg_list, 2):
                    if seg1 < seg2:
                        pair = (seg1, seg2)
                    else:
                        pair = (seg2, seg1)
                    pair_shared_views[pair] += 1
    
    # 填充相似度矩阵
    for pair, count in pair_shared_views.items():
        if count >= required_views:
            seg1, seg2 = pair
            i = segment_to_idx[seg1]
            j = segment_to_idx[seg2]
            similarity_matrix[i, j] = count
            similarity_matrix[j, i] = count
    
    # 如果没有足够的相似性信息，每个线段自成一类
    if np.sum(similarity_matrix) == 0:
        return [{seg} for seg in all_segments]
    
    # 转换为 NetworkX 图（Chinese Whispers 通常用 NetworkX 实现）
    import networkx as nx
    
    G = nx.Graph()
    for i in range(num_segments):
        G.add_node(i)
    
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if similarity_matrix[i, j] > 0:
                G.add_edge(i, j, weight=similarity_matrix[i, j])
    
    # 如果没有边，每个节点自成一类
    if not G.edges():
        return [{seg} for seg in all_segments]
    
    # 定义 Chinese Whispers 聚类函数
    def chinese_whispers(graph, iterations=20):
        # 初始化每个节点的唯一标签
        for node in graph.nodes():
            graph.nodes[node]['label'] = node
        
        # 迭代更新标签
        for _ in range(iterations):
            nodes = list(graph.nodes())
            np.random.shuffle(nodes)  # 随机顺序更新
            
            for node in nodes:
                neighbors = list(graph.neighbors(node))
                if not neighbors:
                    continue
                
                # 统计邻居标签及其权重
                label_weights = {}
                for neighbor in neighbors:
                    label = graph.nodes[neighbor]['label']
                    weight = graph[node][neighbor].get('weight', 1.0)
                    label_weights[label] = label_weights.get(label, 0.0) + weight
                
                # 选择权重最大的标签
                if label_weights:
                    max_label = max(label_weights.items(), key=lambda x: x[1])[0]
                    graph.nodes[node]['label'] = max_label
        
        # 收集聚类结果
        clusters = {}
        for node in graph.nodes():
            label = graph.nodes[node]['label']
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)
        
        return clusters.values()
    
    # 执行 Chinese Whispers 聚类
    communities = chinese_whispers(G)
    
    # 收集聚类结果
    clusters = {}
    for cluster_id, node_ids in enumerate(communities):
        for node_id in node_ids:
            seg = idx_to_segment[node_id]
            clusters.setdefault(cluster_id, set()).add(seg)
    
    return clusters



def Leiden_community(mask_assignments, required_views=3, n_clusters=None):
    """
    使用Leiden算法对3D线段进行聚类
    
    参数:
        mask_assignments: 一个字典，键是视角名称，值是字典{线段索引: mask_id}
        required_views: 需要多少个视角下mask相同才认为属于同一类
        n_clusters: 聚类数量，如果为None则自动确定（Leiden算法通常自动确定）
        
    返回:
        一个列表，包含所有聚类结果，每个聚类是一个线段索引的集合
    """
    if not mask_assignments:
        return []
    
    # 收集所有线段索引
    all_segments = set()
    for view_data in mask_assignments.values():
        all_segments.update(view_data.keys())
    all_segments = sorted(all_segments)
    
    # 创建线段索引到连续整数的映射
    segment_to_idx = {seg: idx for idx, seg in enumerate(all_segments)}
    idx_to_segment = {idx: seg for seg, idx in segment_to_idx.items()}
    num_segments = len(segment_to_idx)
    
    # 如果没有线段，返回空结果
    if num_segments == 0:
        return []
    
    # 构建相似度矩阵
    similarity_matrix = np.zeros((num_segments, num_segments))
    
    from itertools import combinations
    from collections import defaultdict
    
    # 创建线段对到共享视角数的映射
    pair_shared_views = defaultdict(int)
    
    # 对于每个视角，记录哪些mask对应哪些线段
    for view_data in mask_assignments.values():
        # 创建mask到线段列表的映射
        mask_to_segments = defaultdict(list)
        for seg, mask in view_data.items():
            mask_to_segments[mask].append(seg)
        
        # 对每个mask下的所有线段对增加共享计数
        for seg_list in mask_to_segments.values():
            if len(seg_list) > 1:
                for seg1, seg2 in combinations(seg_list, 2):
                    if seg1 < seg2:
                        pair = (seg1, seg2)
                    else:
                        pair = (seg2, seg1)
                    pair_shared_views[pair] += 1
    
    # 填充相似度矩阵
    for pair, count in pair_shared_views.items():
        if count >= required_views:
            seg1, seg2 = pair
            i = segment_to_idx[seg1]
            j = segment_to_idx[seg2]
            similarity_matrix[i, j] = count
            similarity_matrix[j, i] = count
    
    # 如果没有足够的相似性信息，每个线段自成一类
    if np.sum(similarity_matrix) == 0:
        return [{seg} for seg in all_segments]
    
    # 转换为igraph图对象
    import igraph as ig
    
    # 创建图
    edges = []
    weights = []
    for i in range(num_segments):
        for j in range(i+1, num_segments):
            if similarity_matrix[i, j] > 0:
                edges.append((i, j))
                weights.append(similarity_matrix[i, j])
    
    # 如果没有边，每个节点自成一类
    if not edges:
        return [{seg} for seg in all_segments]
    
    # 创建图
    g = ig.Graph()
    g.add_vertices(num_segments)
    g.add_edges(edges)
    g.es['weight'] = weights
    
    # 使用Leiden算法进行社区发现
    partition = g.community_leiden(
        objective_function='modularity',
        weights='weight',
        resolution_parameter=1.0,
        n_iterations=-1  # 直到收敛
    )
    
    # 收集聚类结果
    clusters = {}
    for idx, label in enumerate(partition.membership):
        seg = idx_to_segment[idx]
        clusters.setdefault(label, set()).add(seg)
    
    return clusters


def Node2Vec_HDBSCAN(mask_assignments, required_views=3, n_clusters=None):
    """
    使用 Node2Vec + HDBSCAN 对 3D 线段进行聚类
    
    参数:
        mask_assignments: 字典 {视角: {线段索引: mask_id}}
        required_views: 最少共现视角数
        n_clusters: 保留参数（HDBSCAN自动确定）
        
    返回:
        聚类结果列表，每个聚类是线段索引的集合
    """
    # === 1. 数据准备 ===
    all_segments = set()
    for view_data in mask_assignments.values():
        all_segments.update(view_data.keys())
    all_segments = sorted(all_segments)
    
    segment_to_idx = {seg: idx for idx, seg in enumerate(all_segments)}
    idx_to_segment = {idx: seg for seg, idx in segment_to_idx.items()}
    
    if not segment_to_idx:
        return []

    # === 2. 构建加权图 ===
    G = nx.Graph()
    for seg in all_segments:
        G.add_node(segment_to_idx[seg])
    
    # 计算共现权重
    co_occurrence = defaultdict(int)
    for view_data in mask_assignments.values():
        mask_to_segments = defaultdict(list)
        for seg, mask in view_data.items():
            mask_to_segments[mask].append(seg)
        
        for seg_list in mask_to_segments.values():
            for seg1, seg2 in combinations(seg_list, 2):
                if seg1 < seg2:
                    pair = (seg1, seg2)
                else:
                    pair = (seg2, seg1)
                co_occurrence[pair] += 1
    
    # 添加边（仅保留满足阈值的连接）
    for (seg1, seg2), count in co_occurrence.items():
        if count >= required_views:
            G.add_edge(segment_to_idx[seg1], segment_to_idx[seg2], weight=count)
    
    if not G.edges():
        return [{seg} for seg in all_segments]

    # === 3. Node2Vec 嵌入 ===
    node2vec = Node2Vec(
        G,
        dimensions=64,       # 嵌入维度
        walk_length=30,       # 随机游走长度
        num_walks=200,        # 每个节点的游走次数
        weight_key='weight',  # 使用边权重
        workers=4            # 并行线程
    )
    
    model = node2vec.fit(window=10, min_count=1)
    embeddings = np.array([model.wv[str(i)] for i in range(len(all_segments))])

    # === 4. HDBSCAN 聚类 ===
    # 标准化嵌入向量
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    
    clusterer = HDBSCAN(
        min_cluster_size=5,          # 最小簇大小
        min_samples=3,               # 核心点所需邻居数
        cluster_selection_method='eom',  # 选择聚类方法
        metric='euclidean'           # 距离度量
    )
    clusterer.fit(X_scaled)
    
    # === 5. 整理结果 ===
    clusters = defaultdict(set)
    for idx, label in enumerate(clusterer.labels_):
        if label != -1:  # 过滤噪声点（HDBSCAN将噪声标记为-1）
            seg = idx_to_segment[idx]
            clusters[label].add(seg)
    
    # 将噪声点各自作为单独聚类
    noise_indices = np.where(clusterer.labels_ == -1)[0]
    for noise_idx in noise_indices:
        seg = idx_to_segment[noise_idx]
        clusters[len(clusters)] = {seg}

    # 可视化嵌入空间（可选）
    if True:  # 设为True可启用可视化
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(n_components=2)
        X_2d = tsne.fit_transform(embeddings)

        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusterer.labels_, cmap='Spectral', s=5)
        plt.colorbar()
        plt.title("Node2Vec Embeddings with HDBSCAN Clustering")
        plt.show()
    
    return clusters


def similarity_HDBCAN(all_line3d_to_mask, required_views=3, min_cluster_size=5):
    """
    基于相似度矩阵直接进行 HDBSCAN 聚类
    
    参数:
        all_line3d_to_mask: 字典 {line3d_id: {cam_id: mask_id}}
        required_views: 最少共现视角数
        min_cluster_size: HDBSCAN 最小簇大小
        
    返回:
        聚类结果列表，每个聚类是线段索引的集合
    """
    # === 1. 数据准备 ===
    all_segments = sorted(all_line3d_to_mask.keys())
    segment_to_idx = {seg: idx for idx, seg in enumerate(all_segments)}
    idx_to_segment = {idx: seg for seg, idx in segment_to_idx.items()}
    
    if not segment_to_idx:
        return []

    # === 2. 构建相似度矩阵 ===
    n = len(all_segments)
    similarity_matrix = np.zeros((n, n))
    
    # 按相机组织数据
    cam_to_mask_segments = defaultdict(lambda: defaultdict(list))
    for line3d_id, cam_mask_dict in all_line3d_to_mask.items():
        for cam_id, mask_id in cam_mask_dict.items():
            cam_to_mask_segments[cam_id][mask_id].append(line3d_id)
    
    # 填充相似度矩阵
    for cam_id, mask_segments in cam_to_mask_segments.items():
        for seg_list in mask_segments.values():
            for seg1, seg2 in combinations(seg_list, 2):
                i, j = segment_to_idx[seg1], segment_to_idx[seg2]
                similarity_matrix[i, j] += 1
                similarity_matrix[j, i] += 1
    
    # 应用共现阈值
    similarity_matrix[similarity_matrix < required_views] = 0
    
    # 如果没有有效连接，每个线段自成一类
    if np.sum(similarity_matrix) == 0:
        return [{seg} for seg in all_segments]

    # === 3. 转换为距离矩阵（HDBSCAN需要）===
    # 相似度 -> 距离（相似度越高，距离越小）
    max_sim = np.max(similarity_matrix)
    distance_matrix = max_sim - similarity_matrix + 1e-6  # 避免零距离
    
    # === 4. HDBSCAN 聚类 ===
    clusterer = HDBSCAN(
        metric='precomputed',          # 使用预计算距离矩阵
        min_cluster_size=min_cluster_size,
        min_samples=1,                # 核心点所需最小邻居数
        cluster_selection_method='eom'  # 聚类选择方法
    )
    clusterer.fit(distance_matrix)
    
    # === 5. 整理结果 ===
    clusters = defaultdict(set)
    for idx, label in enumerate(clusterer.labels_):
        seg = idx_to_segment[idx]
        if label != -1:
            clusters[label].add(seg)
        else:
            # 噪声点单独成簇
            clusters[len(clusters)] = {seg}
    
    return clusters


def Leiden_community_normalized(mask_assignments, required_views=3, n_clusters=None):
    """
    使用Leiden算法对3D线段进行聚类
    
    参数:
        mask_assignments: 一个字典，键是视角名称，值是字典{线段索引: mask_id}
        required_views: 需要多少个视角下mask相同才认为属于同一类
        n_clusters: 聚类数量，如果为None则自动确定（Leiden算法通常自动确定）
        
    返回:
        一个列表，包含所有聚类结果，每个聚类是一个线段索引的集合
    """
    if not mask_assignments:
        return []
    
    # 收集所有线段索引
    all_segments = set()
    for view_data in mask_assignments.values():
        all_segments.update(view_data.keys())
    all_segments = sorted(all_segments)
    
    # 创建线段索引到连续整数的映射
    segment_to_idx = {seg: idx for idx, seg in enumerate(all_segments)}
    idx_to_segment = {idx: seg for seg, idx in segment_to_idx.items()}
    num_segments = len(segment_to_idx)
    
    # 如果没有线段，返回空结果
    if num_segments == 0:
        return []
    
    # 构建相似度矩阵
    similarity_matrix = np.zeros((num_segments, num_segments))
    
    from itertools import combinations
    from collections import defaultdict
    
    # 创建线段对到共享视角数的映射
    pair_shared_views = defaultdict(int)
    # 创建线段对到共同可见的视角数的映射
    pair_common_views = defaultdict(int)
    
    # 对于每个视角，记录哪些mask对应哪些线段
    for view_name, view_data in mask_assignments.items():
        # 创建mask到线段列表的映射
        mask_to_segments = defaultdict(list)
        for seg, mask in view_data.items():
            mask_to_segments[mask].append(seg)
        
        # 对每个mask下的所有线段对增加共享计数
        for seg_list in mask_to_segments.values():
            if len(seg_list) > 1:
                for seg1, seg2 in combinations(seg_list, 2):
                    if seg1 < seg2:
                        pair = (seg1, seg2)
                    else:
                        pair = (seg2, seg1)
                    pair_shared_views[pair] += 1
        
        # 更新每对线段的共同可见视角数
        segments_in_view = set(view_data.keys())
        for seg1, seg2 in combinations(segments_in_view, 2):
            if seg1 < seg2:
                pair = (seg1, seg2)
            else:
                pair = (seg2, seg1)
            pair_common_views[pair] += 1
    
    # 填充相似度矩阵（归一化后的相似度）
    for pair, count in pair_shared_views.items():
        if count >= required_views:
            seg1, seg2 = pair
            i = segment_to_idx[seg1]
            j = segment_to_idx[seg2]
            # 计算归一化相似度：共享视角数 / 共同可见视角数
            common_views = pair_common_views.get(pair, 1)  # 避免除以0
            normalized_similarity = count / common_views
            similarity_matrix[i, j] = normalized_similarity
            similarity_matrix[j, i] = normalized_similarity
    
    # 如果没有足够的相似性信息，每个线段自成一类
    if np.sum(similarity_matrix) == 0:
        return [{seg} for seg in all_segments]
    
    # 转换为igraph图对象
    import igraph as ig
    
    # 创建图
    edges = []
    weights = []
    for i in range(num_segments):
        for j in range(i+1, num_segments):
            if similarity_matrix[i, j] > 0:
                edges.append((i, j))
                weights.append(similarity_matrix[i, j])
    
    # 如果没有边，每个节点自成一类
    if not edges:
        return [{seg} for seg in all_segments]
    
    # 创建图
    g = ig.Graph()
    g.add_vertices(num_segments)
    g.add_edges(edges)
    g.es['weight'] = weights
    
    # 使用Leiden算法进行社区发现
    partition = g.community_leiden(
        objective_function='modularity',
        weights='weight',
        resolution_parameter=1.0,
        n_iterations=-1  # 直到收敛
    )
    
    # 收集聚类结果
    clusters = {}
    for idx, label in enumerate(partition.membership):
        seg = idx_to_segment[idx]
        clusters.setdefault(label, set()).add(seg)
    
    return clusters