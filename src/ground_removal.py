import numpy as np
import open3d as o3d


# ==============================
# 1. RANSAC 지면 제거
# ==============================

def remove_ground(xyz, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    ground = xyz[inliers]
    mask = np.ones(len(xyz), dtype=bool)
    mask[inliers] = False
    obstacles = xyz[mask]

    return ground, obstacles, plane_model


# ==============================
# 2. 지면 비율(%) 계산
# ==============================

def ground_ratio(xyz, **kwargs):
    ground, obstacles, _ = remove_ground(xyz, **kwargs)
    return 100.0 * len(ground) / len(xyz)
