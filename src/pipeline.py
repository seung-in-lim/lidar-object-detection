import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from load_pointcloud import load_velodyne_bin, get_xyz
from ground_removal import remove_ground
from clustering import cluster_obstacles, cluster_stats
from detection import clusters_to_boxes


# ==============================
# 1. 전체 파이프라인 실행
# ==============================

def run_pipeline(bin_path, ground_threshold=0.2, eps=0.8, min_points=15, verbose=True):
    points = load_velodyne_bin(bin_path)
    xyz = get_xyz(points)

    ground, obstacles, plane = remove_ground(xyz, distance_threshold=ground_threshold)

    labels = cluster_obstacles(obstacles, eps=eps, min_points=min_points)
    n_clusters, n_noise = cluster_stats(labels)

    boxes, rejected = clusters_to_boxes(obstacles, labels)

    if verbose:
        print(f"[{bin_path}]")
        print(f"  전체 점: {len(xyz):,} | 지면: {len(ground):,} "
              f"({100*len(ground)/len(xyz):.1f}%) | 장애물: {len(obstacles):,}")
        print(f"  군집: {n_clusters} | 노이즈: {n_noise:,}")
        print(f"  최종 검출: {len(boxes)} "
              f"(제외 - 크기:{rejected['size']} 높이:{rejected['height']} 형상:{rejected['shape']})")

    return {
        "xyz": xyz, "ground": ground, "obstacles": obstacles,
        "labels": labels, "boxes": boxes, "rejected": rejected,
    }


# ==============================
# 2. 검출 결과 시각화 (BEV)
# ==============================

def visualize(result, title="LiDAR Obstacle Detection", save_path=None):
    obstacles = result["obstacles"]
    boxes = result["boxes"]

    fig, ax = plt.subplots(figsize=(13, 12))
    ax.scatter(obstacles[:, 0], obstacles[:, 1], c="lightsteelblue", s=0.5)
    for b in boxes:
        rect = patches.Rectangle(
            (b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
            linewidth=1.8, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
    ax.plot(0, 0, "k*", markersize=18)
    ax.set_title(f"{title} - {len(boxes)} objects")
    ax.set_xlabel("x: forward (m)")
    ax.set_ylabel("y: left (m)")
    ax.axis("equal")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장됨: {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LiDAR 장애물 검출 파이프라인")
    parser.add_argument("bin_path", help="KITTI .bin 라이다 파일 경로")
    parser.add_argument("--save", default=None, help="결과 이미지 저장 경로")
    parser.add_argument("--ground_threshold", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.8)
    parser.add_argument("--min_points", type=int, default=15)
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")

    result = run_pipeline(
        args.bin_path,
        ground_threshold=args.ground_threshold,
        eps=args.eps, min_points=args.min_points,
    )
    visualize(result, save_path=args.save)
