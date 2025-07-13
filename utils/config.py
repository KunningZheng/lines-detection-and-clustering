import os 

def get_config():
    """Return configuration parameters for the clustering process."""
    config = {
        'workspace_path': '/home/rylynn/Pictures/Clustering_Workspace',
        'scene_name': 'group_selectByPt_chunk3',
        'k_near': 10,
        'clustering_method': 'leiden'
    }
    print("Configuration parameters:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("==============================================================================")
    return config


class PathManager:
    def __init__(self, workspace_path, scene_name):
        self.workspace_path = workspace_path
        self.scene_name = scene_name
        
    @property
    def sparse_model_path(self):
        return os.path.join(self.workspace_path, self.scene_name, 'Colmap', 'sparse')
    
    @property
    def images_path(self):
        return os.path.join(self.workspace_path, self.scene_name, 'Colmap', 'images')
    
    @property
    def line3dpp_path(self):
        return os.path.join(self.workspace_path, self.scene_name, 'Line3D++')
    
    @property
    def single_mask_path(self):
        return os.path.join(self.workspace_path, self.scene_name, 'SAM_Mask', 'Single_Mask')
    
    @property
    def merged_mask_path(self):
        return os.path.join(self.workspace_path, self.scene_name, 'SAM_Mask', 'Merged_Mask')
    
    @property
    def intermediate_output_path(self):
        path = os.path.join(self.workspace_path, self.scene_name, 'intermediate_outputs')
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_line3d_to_mask_path(self):
        return os.path.join(self.intermediate_output_path, 'all_line3d_to_mask.json')
    
    def get_lines3d_clusters_path(self, clustering_method):
        return os.path.join(self.intermediate_output_path, f'lines3d_clusters_{clustering_method}.json')