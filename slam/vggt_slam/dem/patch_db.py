from __future__ import annotations
import os, json, glob
from typing import Optional, Tuple, Dict
import numpy as np
import cv2

def load_latest_dem_meta(run_root: str) -> Optional[dict]:
    metas = sorted(glob.glob(os.path.join(run_root, "DEMs", "dem_*_meta.json")))
    if not metas:
        return None
    with open(metas[-1], "r") as f:
        return json.load(f)

def load_latest_dem_grid(run_root: str) -> Optional[np.ndarray]:
    npys = sorted(glob.glob(os.path.join(run_root, "DEMs", "dem_*.npy")))
    if not npys:
        return None
    return np.load(npys[-1])  # full grid (nx, ny) float32 meters above plane

def crop_dem_patch_from_bbox(H_full: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    """bbox=(x0,x1,y0,y1) in DEM grid indices -> returns copy float32 patch (>=0 where painted)."""
    x0,x1,y0,y1 = bbox
    x0 = max(0, min(x0, H_full.shape[0]-1))
    x1 = max(0, min(x1, H_full.shape[0]-1))
    y0 = max(0, min(y0, H_full.shape[1]-1))
    y1 = max(0, min(y1, H_full.shape[1]-1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((32,32), np.float32)
    return H_full[x0:x1+1, y0:y1+1].copy().astype(np.float32)

def normalize_patch_for_matching(p: np.ndarray, out_size: int = 128) -> np.ndarray:
    """
    Turn a DEM height patch into a contrast-normalized 8-bit image for matching/embedding.
    Steps: clip to robust max, zscore, minmax to [0,255], resize to (out_size,out_size).
    """
    if p.size == 0:
        return np.zeros((out_size,out_size), np.uint8)
    v = p.copy()
    vmax = np.percentile(v[v>0], 99.0) if np.any(v>0) else 1.0
    v = np.clip(v, 0.0, max(vmax, 1e-6))
    # z-score on positive region
    good = v > 0
    if np.any(good):
        mu = float(v[good].mean()); sd = float(v[good].std() + 1e-6)
        v[good] = (v[good] - mu) / sd
    v = (v - v.min()) / (v.max() - v.min() + 1e-6)
    v8 = (v * 255.0).astype(np.uint8)
    v8 = cv2.resize(v8, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return v8

def canny_edge(v8: np.ndarray) -> np.ndarray:
    """Simple edges; returns uint8 0/255."""
    e = cv2.Canny(v8, 50, 150)
    return e

def ncc_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross correlation on uint8 images (higher=better)."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    am = a.mean(); bm = b.mean()
    an = a - am; bn = b - bm
    denom = np.sqrt((an*an).sum() * (bn*bn).sum()) + 1e-12
    return float((an*bn).sum() / denom)
