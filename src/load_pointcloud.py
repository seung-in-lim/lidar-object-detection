import os
import numpy as np


# ==============================
# 1. KITTI .bin 라이다 파일 읽기
# ==============================

def load_velodyne_bin(bin_path):
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"라이다 파일을 찾을 수 없습니다: {bin_path}")
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


# ==============================
# 2. xyz 좌표만 추출
# ==============================

def get_xyz(points):
    return points[:, :3]


# ==============================
# 3. 포인트클라우드 통계 요약
# ==============================

def summarize(points):
    return {
        "num_points": points.shape[0],
        "x_range": (float(points[:, 0].min()), float(points[:, 0].max())),
        "y_range": (float(points[:, 1].min()), float(points[:, 1].max())),
        "z_range": (float(points[:, 2].min()), float(points[:, 2].max())),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pts = load_velodyne_bin(sys.argv[1])
        info = summarize(pts)
        print(f"점 개수: {info['num_points']:,}")
        print(f"x 범위: {info['x_range'][0]:.1f} ~ {info['x_range'][1]:.1f} m")
        print(f"y 범위: {info['y_range'][0]:.1f} ~ {info['y_range'][1]:.1f} m")
        print(f"z 범위: {info['z_range'][0]:.1f} ~ {info['z_range'][1]:.1f} m")
