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
from tqdm import tqdm
import json

# internal
from datasets.sfm_reader import load_sparse_model
from datasets.overlap_detector import match_pair
from datasets.mask_processor import merge_masks
from datasets.line3dpp_loader import parse_line_segments, parse_lines3dpp
from datasets.lines3d_gt_generator import lines3d_gt_generator
from utils.json2dict import json_decode_with_int, convert_sets
from utils.config import get_config, PathManager
from clustering.mask_association import load_mask_lines_association, associate_projline_to_mask
from clustering.lines_correspondence import LineCorrespondence
from clustering.lines_tools import visualize_line_clusters, determine_line3d_visibility
from clustering.clustering_methods import Chinese_Whispers, Leiden_community, Node2Vec_HDBSCAN, similarity_HDBCAN


clustering_methods = {
    "CW": Chinese_Whispers,
    "leiden": Leiden_community,
    "Node2Vec_HDBSCAN": Node2Vec_HDBSCAN,
    "similarity_HDBCAN": similarity_HDBCAN
}

def apply_clustering_method(Clustering_method, *args, **kwargs):
    try:
        method = clustering_methods[Clustering_method]
        return method(*args, **kwargs)
    except KeyError:
        raise ValueError(f"未知的聚类方法: {Clustering_method}。可选方法: {list(clustering_methods.keys())}")


def load_and_process_data(path_manager, k_near):
    """Load and process all required data for clustering."""
    # Load sparse model
    camerasInfo, points_in_images, points3d_xyz = load_sparse_model(path_manager.sparse_model_path)
    
    # Compute match matrix
    #match_matrix = match_pair(camerasInfo, points_in_images, k_near=k_near)
    
    
    # Process masks
    merge_masks(path_manager.single_mask_path, path_manager.merged_mask_path)
    
    # Associate masks and lines
    all_mask_to_lines, all_line_to_mask = load_mask_lines_association(
        camerasInfo, 
        path_manager.merged_mask_path, 
        path_manager.line3dpp_path, 
        path_manager.intermediate_output_path
    )
    
    # Parse line segments
    lines3d, residuals2d_for_lines3d = parse_lines3dpp(path_manager.line3dpp_path)
    line_corr = LineCorrespondence(residuals2d_for_lines3d)

    # Generate 3D Line instanceID groudtruth
    _, lines2d_instanceID_gt = load_mask_lines_association(
        camerasInfo,
        path_manager.gt_mask_path,
        path_manager.line3dpp_path,
        path_manager.groundtruth_path,
    )
    lines3d_instanceID_gt = lines3d_gt_generator(lines3d, residuals2d_for_lines3d, lines2d_instanceID_gt, 
                                                 path_manager.groundtruth_path, show=True)

    
    return camerasInfo, points3d_xyz, all_line_to_mask, lines3d, line_corr


def associate_line3d_to_mask(camerasInfo, points3d_xyz, all_line_to_mask, lines3d, line_corr, merged_mask_path, output_path):
    """Associate 3D lines with masks across all views."""
    if os.path.exists(output_path):
        print("Load existing lines3d and mask associations file")
        with open(output_path, 'r') as f:
            all_line3d_to_mask = json.load(f)
        # Convert keys to int
        all_line3d_to_mask = {
            int(line3d_id): {int(cam_id): int(mask_id) for cam_id, mask_id in line3d_to_mask.items()}
            for line3d_id, line3d_to_mask in all_line3d_to_mask.items()
        }
        return all_line3d_to_mask
    
    all_line3d_to_mask = {}        
    for cam_id, line_to_mask in tqdm(all_line_to_mask.items(), desc="在各视角下寻找lines3d对应的local mask"):
        cam_id = int(cam_id)
        line3d_to_mask = {}
        
        # Case 1: Direct correspondence via 2D segments
        for seg_id, mask_id in line_to_mask.items():
            line3d_id = line_corr.find_line3d_by_cam_seg(cam_id, int(seg_id))
            if line3d_id is not None:
                line3d_to_mask[line3d_id] = mask_id
        
        # Case 2: Projection-based association
        for line3d_id in tqdm(range(len(lines3d)), desc=f"Processing cam {cam_id}"):
            if line3d_to_mask.get(line3d_id) is not None:
                continue
                
            cam_dict = camerasInfo[cam_id]
            visibility = determine_line3d_visibility(cam_dict, lines3d[line3d_id], points3d_xyz)
            
            if visibility is not False:
                p1_proj, p2_proj = visibility
                mask_id = associate_projline_to_mask(np.array([p1_proj, p2_proj]), cam_dict, merged_mask_path)
                if mask_id != -1:
                    line3d_to_mask[line3d_id] = int(mask_id)
                    
        all_line3d_to_mask[cam_id] = line3d_to_mask
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_line3d_to_mask, f, indent=4)
    
    return all_line3d_to_mask


def main():
    # Load configuration
    config = get_config()
    path_manager = PathManager(config['workspace_path'], config['scene_name'])
    
    # Load and process data
    camerasInfo, points3d_xyz, all_line_to_mask, lines3d, line_corr = load_and_process_data(
        path_manager, config['k_near']
    )
    
    # Associate lines3d with masks
    all_line3d_to_mask = associate_line3d_to_mask(
        camerasInfo, points3d_xyz, all_line_to_mask, 
        lines3d, line_corr, path_manager.merged_mask_path,
        path_manager.get_line3d_to_mask_path()
    )
    
    # Apply clustering
    lines3d_clusters_path = path_manager.get_lines3d_clusters_path(config['clustering_method'])
    if os.path.exists(lines3d_clusters_path):
        with open(lines3d_clusters_path, 'r') as f:
            loaded = json.load(f)
            lines3d_clusters = json_decode_with_int(loaded)
    else:
        lines3d_clusters = apply_clustering_method(
            config['clustering_method'], 
            all_line3d_to_mask, 
            required_views=1
        )
        # Save results if needed
        with open(lines3d_clusters_path, 'w') as f:
            json.dump(convert_sets(lines3d_clusters), f)
    
    print(f"Number of Clusters: {len(lines3d_clusters)}")
    # Filter and visualize results
    filtered_clusters = {
        cluster_id: lines 
        for cluster_id, lines in lines3d_clusters.items() 
        if len(lines) > 1
    }
    print(f"Number of Clusters after filtering: {len(filtered_clusters)}")
    
    # Visualize results
    visualize_line_clusters(lines3d, filtered_clusters)

if __name__ == '__main__':
    main()