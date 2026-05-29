import numpy as np
import open3d as o3d


# ==============================
# 1. DBSCAN 클러스터링
# ==============================

def cluster_obstacles(obstacles, eps=0.8, min_points=15):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obstacles)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return labels


# ==============================
# 2. 군집 개수 / 노이즈 개수
# ==============================

def cluster_stats(labels):
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    n_noise = int((labels == -1).sum())
    return n_clusters, n_noise
