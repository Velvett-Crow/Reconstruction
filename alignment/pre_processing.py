'''Performs alignment processing'''

import open3d as o3d

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample and clean a point cloud."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down


def extract_fpfh_features(pcd, voxel_size):
    """Compute FPFH feature descriptors."""
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )


def find_feature_transform(source_down, target_down, source_fpfh, target_fpfh, voxel_size,
                           ransac_n=3, max_iteration=40000, confidence=500):
    """Coarse alignment using RANSAC feature matching."""
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, confidence)
    )
    return result
