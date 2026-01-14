import open3d as o3d
import numpy as np
import glob
import os
import matplotlib.cm as cm

from pre_processing import (
    preprocess_point_cloud,
    extract_fpfh_features,
    find_feature_transform
)

def compute_information_matrix(source, target, transform, voxel_size):
    """Information matrix for ICP-based edges."""
    return o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, voxel_size * 1.5, transform
    )

def pairwise_registration(source, target, voxel_size, distance_multiplier):
    """RANSAC + ICP pairwise alignment."""
    src_down = source # preprocess_point_cloud(source, voxel_size)
    tgt_down = target # preprocess_point_cloud(target, voxel_size)

    src_fpfh = extract_fpfh_features(src_down, voxel_size)
    tgt_fpfh = extract_fpfh_features(tgt_down, voxel_size)

    # RANSAC coarse alignment
    result_ransac = find_feature_transform(
        src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size
    )

    # ICP fine alignment
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        voxel_size * distance_multiplier,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return result_icp.transformation, result_icp.inlier_rmse

def build_pose_graph(pcds, config, loop_step=5):
    N = len(pcds)
    voxel_size = config["voxel_size"]
    distance_multiplier = config["distance_multiplier"]

    pose_graph = o3d.pipelines.registration.PoseGraph() # store and manage pose graphs
    odometry = np.eye(4) # identity matrix for the nodes
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for i in range(N):
        for j in range(i+1, N):

            # Only check near neighbors + loop closures
            if j == i + 1 or (j % loop_step == 0 and abs(i - j) > 3):
                
                Tij, rmse = pairwise_registration(
                    pcds[i], pcds[j], voxel_size, distance_multiplier
                )
                info = compute_information_matrix(pcds[i], pcds[j], Tij, voxel_size)

                if j == i + 1:
                    # odometry edge
                    odometry = odometry @ Tij
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                    )
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i, j, Tij, info, uncertain=False
                        )
                    )
                else:
                    # loop closure
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            i, j, Tij, info, uncertain=True
                        )
                    )
    return pose_graph

def global_optimize(pose_graph):
    # configure the optimization tool
    config_opt = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.0025,
        edge_prune_threshold=0.5,
        reference_node=0
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        config_opt
    )

def main():
    config = {
        "voxel_size": 0.0012,
        "distance_multiplier": 2
    }

    # Load point clouds
    file_list = sorted(glob.glob("/home/jovyan/videos_renders/point_clouds/dy/resized/original/global/*.ply"))
    pcds = [o3d.io.read_point_cloud(f) for f in file_list]

    # estimate normals for all clouds
    print("Estimating normals for all point clouds")
    for p in pcds:
        p.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=config["voxel_size"] * 2,
                max_nn=30
            )
        )

    print("Building Pose Graph")
    pose_graph = build_pose_graph(pcds, config, loop_step=1)

    print("Running Global Optimization")
    global_optimize(pose_graph)
    '''
    print("Applying optimized transforms")
    pcds_opt = []
    for p, node in zip(pcds, pose_graph.nodes):
        p_transformed = p.transform(node.pose)
        pcds_opt.append(p_transformed)
    '''
    print("Applying optimized transforms")
    pcds_opt = []
    for p, node in zip(pcds, pose_graph.nodes):
        p.transform(node.pose)
        pcds_opt.append(p)

    # final ICP refinement
    for i in range(len(pcds_opt) - 1):
        result = o3d.pipelines.registration.registration_icp(
            pcds_opt[i],
            pcds_opt[i + 1],
            0.0024,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        pcds_opt[i].transform(result.transformation)
    '''
    cmap = cm.get_cmap("tab20", len(pcds_opt))

    for i, pcd in enumerate(pcds_opt):
        r, g, b, _ = cmap(i)
        pcd.paint_uniform_color([r, g, b])
    
    print("Visualizing result")
    pcds_opt = [pcd.voxel_down_sample(voxel_size=0.001) for pcd in pcds_opt]
    # pcds_opt = pcds_opt.voxel_down_sample(voxel_size=0.05)
    fig = o3d.visualization.draw_plotly(pcds_opt)
    '''
    
    
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in pcds_opt:
        merged_pcd += pcd
        
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.001)
    merged_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.003,
            max_nn=30
        )
    )
    merged_pcd.orient_normals_consistent_tangent_plane(k=30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        merged_pcd,
        depth=9
    )
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)

    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    fig = o3d.visualization.draw_plotly([mesh])
    

if __name__ == "__main__":
    main()
