import torch
import numpy as np
import open3d as o3d
from gedi import GeDi  # Assuming GeDi class and its dependencies are correctly defined elsewhere
from time import perf_counter
from tqdm import tqdm  # Import tqdm for progress bar functionality
from icecream import ic
# Configuration for GeDi
config = {
    'dim': 32,
    'samples_per_batch': 500,
    'samples_per_patch_lrf': 4000,
    'samples_per_patch_out': 512,
    'r_lrf': 0.5,
    'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'
}

voxel_size = 0.01
patches_per_pair = 100000
num_point_clouds = 32  # You have 32 point clouds

# Initialize GeDi
gedi = GeDi(config=config)

# Loop through each pair of point clouds with tqdm for progress visualization
for i in tqdm(range(15, num_point_clouds), desc="Processing point cloud pairs"):
    start_time = perf_counter()
    # Load point clouds
    pcd0 = o3d.io.read_point_cloud(f'data/pc{i}.ply')
    pcd1 = o3d.io.read_point_cloud(f'data/pc{i+1}.ply')

    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])

    pcd0.estimate_normals()
    pcd1.estimate_normals()

    # Randomly sampling points
    inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=False)
    inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches_per_pair, replace=False)

    pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
    pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

    # Voxelisation
    pcd0 = pcd0.voxel_down_sample(voxel_size)
    pcd1 = pcd1.voxel_down_sample(voxel_size)

    _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
    _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

    # Computing descriptors
    pcd0_desc = gedi.compute(pts=pts0, pcd=_pcd0)
    pcd1_desc = gedi.compute(pts=pts1, pcd=_pcd1)

    # Preparing for RANSAC
    pcd0_dsdv = o3d.pipelines.registration.Feature()
    pcd1_dsdv = o3d.pipelines.registration.Feature()
    pcd0_dsdv.data = pcd0_desc.T
    pcd1_dsdv.data = pcd1_desc.T

    _pcd0 = o3d.geometry.PointCloud()
    _pcd1 = o3d.geometry.PointCloud()
    _pcd0.points = o3d.utility.Vector3dVector(pts0)
    _pcd1.points = o3d.utility.Vector3dVector(pts1)

    # RANSAC Registration
    est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        _pcd0, _pcd1, pcd0_dsdv, pcd1_dsdv, mutual_filter=True, max_correspondence_distance=2.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4, checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(2.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
    )
    ic(f"RANSAC Fitness: {est_result01.fitness:.2f}")
    ic(i, "iteration ended", perf_counter())
    ic(f'Matching Status: {"Matched" if est_result01.fitness > 0.70 else "Not Matched"}\n')

    # Write registration info
    with open(f'./report/registration_info_pair{i}_{i+1}.txt', 'w') as file:
        file.write(f'Matching Status: {"Matched" if est_result01.fitness > 0.1 else "Not Matched"}\n')
        file.write('Transformation Matrix:\n')
        file.write(str(est_result01.transformation))
        file.write(f'\nFitness Score: {est_result01.fitness}\n')
        file.write(f'Inlier Ratio: {est_result01.inlier_rmse}\n')

    # Apply transformation
    pcd0.transform(est_result01.transformation)
    # Save the transformed point cloud
    o3d.io.write_point_cloud(f'results/transformed_pcd{i}.ply', pcd0)
