import numpy as np


# ==============================
# 군집 → 바운딩 박스 + 후처리 필터
# ==============================

def clusters_to_boxes(obstacles, labels,
                      size_min=0.3, size_max=8.0,
                      height_min=0.4, aspect_max=5.0):
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    boxes = []
    rejected = {"size": 0, "height": 0, "shape": 0}

    for cid in range(n_clusters):
        pts = obstacles[labels == cid]
        size_x = pts[:, 0].max() - pts[:, 0].min()
        size_y = pts[:, 1].max() - pts[:, 1].min()
        size_z = pts[:, 2].max() - pts[:, 2].min()

        # 크기 필터
        if not (size_min < size_x < size_max and size_min < size_y < size_max):
            rejected["size"] += 1
            continue
        # 높이 필터 (지면 잔여물 제거)
        if size_z < height_min:
            rejected["height"] += 1
            continue
        # 형상 필터 (벽 제거)
        aspect = max(size_x, size_y) / (min(size_x, size_y) + 1e-6)
        if aspect > aspect_max:
            rejected["shape"] += 1
            continue

        boxes.append({
            "xmin": float(pts[:, 0].min()), "ymin": float(pts[:, 1].min()),
            "xmax": float(pts[:, 0].max()), "ymax": float(pts[:, 1].max()),
            "n_points": int(len(pts)),
        })

    return boxes, rejected
