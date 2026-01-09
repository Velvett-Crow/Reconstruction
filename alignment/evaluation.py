'''
Evaluating the quality of the point cloud alignment
'''

import open3d as o3d
import numpy as np

class RegistrationEvaluator:
    def __init__(self, noise_bound=0.05):
        self.noise_bound = noise_bound

    def evaluate(self, source_pcd, target_pcd, transformation):
        source_copy = o3d.geometry.PointCloud(source_pcd) # copy to avoid affecting the original
        source_aligned = source_copy.transform(transformation)

        distances = np.asarray(source_aligned.compute_point_cloud_distance(target_pcd))
        rmse = np.sqrt(np.mean(np.square(distances)))
        inlier_mask = distances < self.noise_bound
        inlier_ratio = np.mean(inlier_mask)
        
        return {
            'rmse': rmse,
            'overlap_ratio': inlier_ratio
        }
