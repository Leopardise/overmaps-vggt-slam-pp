from __future__ import annotations
import os
from pathlib import Path
import numpy as np

def load_all_submap_points(submaps_root: str) -> np.ndarray:
    submaps_root = str(submaps_root)
    pts_all = []
    for sm_dir in sorted(Path(submaps_root).glob("sm_*")):
        p = sm_dir / "points_world.npy"
        if p.is_file():
            arr = np.load(str(p)).astype(np.float32).reshape(-1, 3)
            if arr.size:
                pts_all.append(arr)
    if not pts_all:
        return np.zeros((0, 3), np.float32)
    return np.concatenate(pts_all, axis=0)

def sample_gt_mesh(mesh_path: str, n: int = 1_000_000) -> np.ndarray:
    """
    Samples points from a mesh using Open3D.
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise ImportError("Open3D is required for mesh sampling. Install open3d.") from e

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"GT mesh is empty or unreadable: {mesh_path}")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=int(n))
    return np.asarray(pcd.points, dtype=np.float32)

def compute_chamfer(pred_pts: np.ndarray, gt_pts: np.ndarray, icp_thresh: float = 0.05) -> dict:
    """
    Simple symmetric Chamfer using nearest-neighbor distances.
    Optional: can ICP-align first if needed later.
    """
    pred = np.asarray(pred_pts, np.float32).reshape(-1, 3)
    gt = np.asarray(gt_pts, np.float32).reshape(-1, 3)

    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return {"chamfer": float("nan"), "pred_to_gt": float("nan"), "gt_to_pred": float("nan")}

    # KDTree via scipy if available, else sklearn
    try:
        from scipy.spatial import cKDTree as KDTree
        tree_gt = KDTree(gt)
        tree_pr = KDTree(pred)
        d1, _ = tree_gt.query(pred, k=1)
        d2, _ = tree_pr.query(gt, k=1)
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        nn_gt = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(gt)
        nn_pr = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(pred)
        d1, _ = nn_gt.kneighbors(pred, return_distance=True)
        d2, _ = nn_pr.kneighbors(gt, return_distance=True)
        d1 = d1[:, 0]
        d2 = d2[:, 0]

    pred_to_gt = float(np.mean(d1**2))
    gt_to_pred = float(np.mean(d2**2))
    chamfer = float(pred_to_gt + gt_to_pred)

    return {
        "pred_points": int(pred.shape[0]),
        "gt_points": int(gt.shape[0]),
        "pred_to_gt_m2": pred_to_gt,
        "gt_to_pred_m2": gt_to_pred,
        "chamfer_m2": chamfer,
    }
