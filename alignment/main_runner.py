'''Used for pairwise registration'''

import open3d as o3d
import numpy as np
from pre_processing import preprocess_point_cloud, extract_fpfh_features, find_feature_transform
from evaluation import RegistrationEvaluator


def main():
    # Configuration
    config = {
        'voxel_size': 0.01,     # Controls downsampling and feature radius
        'distance_multiplier': 1.5,  # Threshold scaling for RANSAC/ICP
        'noise_bound': 0.00001     # For evaluation
    }

    # Load point clouds
    source_pcd = o3d.io.read_point_cloud("/home/jovyan/videos_renders/point_clouds/dy/resized/rectified/frame_0322.ply")
    target_pcd = o3d.io.read_point_cloud("/home/jovyan/videos_renders/point_clouds/dy/resized/rectified/frame_0323.ply")

    source_down = source_pcd #preprocess_point_cloud(source_pcd, config['voxel_size'])
    target_down = target_pcd #preprocess_point_cloud(target_pcd, config['voxel_size'])

    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=config['voxel_size'] * 2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=config['voxel_size'] * 2, max_nn=30))
    
    print("Extracting FPFH features...")
    source_fpfh = extract_fpfh_features(source_down, config['voxel_size'])
    target_fpfh = extract_fpfh_features(target_down, config['voxel_size'])

    print("Running RANSAC coarse alignment...")
    result_ransac = find_feature_transform(
        source_down, target_down, source_fpfh, target_fpfh, config['voxel_size'], 
        ransac_n=3, max_iteration=40000, confidence=500
    )
    transformation_ransac = result_ransac.transformation
    
    print("Estimating normals for ICP...")
    target_pcd.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=config['voxel_size'] * 2, max_nn=30))

    print("Running ICP fine alignment...")
    result_icp = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        config['voxel_size'] * config['distance_multiplier'],
        transformation_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    transformation_icp = result_icp.transformation

    # Evaluate final registration
    evaluator = RegistrationEvaluator(config['noise_bound'])
    metrics = evaluator.evaluate(source_pcd, target_pcd, transformation_icp)

    print("\n=== Registration Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("============================")

    # Visualize
    source_aligned = source_pcd.transform(transformation_icp)
    # source_aligned.paint_uniform_color([1, 0, 0])
    # target_pcd.paint_uniform_color([0, 1, 0])
    # source_aligned = source_aligned.voxel_down_sample(voxel_size=0.005)
    # target_pcd = target_pcd.voxel_down_sample(voxel_size=0.005)
    merged = source_aligned + target_pcd
    merged.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    merged.orient_normals_consistent_tangent_plane(k=30)
    pcdmesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merged, depth=9)
    # fig = o3d.visualization.draw_plotly([pcdmesh])
    fig = o3d.visualization.draw_plotly([pcdmesh])
    
if __name__ == "__main__":
    main()
