from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree


def write_ply_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    n = points.shape[0]
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, rgb):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def write_ply_mesh(path: Path, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray | None) -> None:
    n = verts.shape[0]
    rgb = None
    if colors is not None and len(colors) == n:
        rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for i, p in enumerate(verts):
            if rgb is None:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            else:
                c = rgb[i]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def voxel_downsample(points: np.ndarray, colors: np.ndarray, conf: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, colors, conf
    keys = np.floor(points / max(voxel, 1e-3)).astype(np.int32)
    # hash voxels
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    out_p = np.zeros((len(uniq), 3), dtype=np.float32)
    out_c = np.zeros((len(uniq), 3), dtype=np.float32)
    out_k = np.zeros((len(uniq),), dtype=np.float32)
    counts = np.bincount(inv)
    np.add.at(out_p, inv, points)
    np.add.at(out_c, inv, colors)
    np.add.at(out_k, inv, conf)
    counts = np.maximum(counts, 1)[:, None]
    out_p /= counts
    out_c /= counts
    out_k /= counts[:, 0]
    return out_p, np.clip(out_c, 0, 1), out_k


def statistical_filter(points: np.ndarray, colors: np.ndarray, conf: np.ndarray, nb: int, std_ratio: float):
    if points.shape[0] < nb + 5:
        return points, colors, conf
    tree = cKDTree(points)
    d, _ = tree.query(points, k=min(nb + 1, points.shape[0]))
    mean_d = d[:, 1:].mean(axis=1)
    mu, sigma = mean_d.mean(), mean_d.std()
    keep = mean_d < mu + std_ratio * sigma
    return points[keep], colors[keep], conf[keep]


def terrain_mesh(points: np.ndarray, colors: np.ndarray, sample_n: int, max_edge: float):
    if points.shape[0] < 8:
        return None, None, None
    rng = np.random.default_rng(0)
    if points.shape[0] > sample_n:
        idx = rng.choice(points.shape[0], sample_n, replace=False)
        pts = points[idx]
        cols = colors[idx]
    else:
        pts, cols = points, colors
    xy = pts[:, :2]
    try:
        tri = Delaunay(xy)
    except Exception:
        return None, None, None
    faces = tri.simplices
    edges = np.linalg.norm(pts[faces[:, [0, 1, 2]]] - pts[faces[:, [1, 2, 0]]], axis=2)
    keep = edges.max(axis=1) < max_edge
    faces = faces[keep]
    if faces.shape[0] == 0:
        return None, None, None
    return pts.astype(np.float32), faces.astype(np.int32), cols.astype(np.float32)


def raster_dsm(points: np.ndarray, resolution: float = 0.5) -> tuple[np.ndarray, tuple[float, float], float]:
    if points.shape[0] == 0:
        return np.zeros((1, 1), np.float32), (0.0, 0.0), resolution
    min_e, min_n = points[:, 0].min(), points[:, 1].min()
    max_e, max_n = points[:, 0].max(), points[:, 1].max()
    w = max(int(np.ceil((max_e - min_e) / resolution)) + 1, 2)
    h = max(int(np.ceil((max_n - min_n) / resolution)) + 1, 2)
    dsm = np.full((h, w), np.nan, dtype=np.float32)
    ix = np.clip(((points[:, 0] - min_e) / resolution).astype(int), 0, w - 1)
    iy = np.clip(((points[:, 1] - min_n) / resolution).astype(int), 0, h - 1)
    for x, y, z in zip(ix, iy, points[:, 2]):
        if np.isnan(dsm[y, x]) or z > dsm[y, x]:
            dsm[y, x] = z
    return dsm, (float(min_e), float(min_n)), resolution

def raster_ortho(points: np.ndarray, colors: np.ndarray, resolution: float = 0.5):
    """Generates a 2D Orthomosaic RGB image from the point cloud."""
    from PIL import Image
    if points.shape[0] == 0:
        return Image.new("RGB", (100, 100), (0, 0, 0))
        
    min_e, min_n = points[:, 0].min(), points[:, 1].min()
    max_e, max_n = points[:, 0].max(), points[:, 1].max()
    w = max(int(np.ceil((max_e - min_e) / resolution)) + 1, 2)
    h = max(int(np.ceil((max_n - min_n) / resolution)) + 1, 2)
    
    # Track Max Z to determine which pixel is visible from top-down
    z_buffer = np.full((h, w), -np.inf, dtype=np.float32)
    img_arr = np.zeros((h, w, 3), dtype=np.uint8)
    
    ix = np.clip(((points[:, 0] - min_e) / resolution).astype(int), 0, w - 1)
    iy = np.clip(((points[:, 1] - min_n) / resolution).astype(int), 0, h - 1)
    
    c_bytes = (colors * 255).astype(np.uint8)
    
    for x, y, z, c in zip(ix, iy, points[:, 2], c_bytes):
        if z > z_buffer[y, x]:
            z_buffer[y, x] = z
            img_arr[y, x] = c
            
    # Flip Y because numpy arrays have Y=0 at the top, but ENU has Y (North) up
    img_arr = np.flipud(img_arr)
    return Image.fromarray(img_arr)
