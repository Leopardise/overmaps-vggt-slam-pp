from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import json, math
import numpy as np

@dataclass
class GlobalFrame:
    # plane & frame
    N: np.ndarray
    d: float
    R: np.ndarray
    o: np.ndarray
    # raster geometry
    mpp: float
    tile_px: int
    nx: int
    ny: int
    u0: float
    v0: float
    # viz scale (percentiles) if present
    clip_lo: float = 1.0
    clip_hi: float = 99.0

def _get(d, *keys, default=None):
    for k in keys:
        if k in d: return d[k]
    return default

def load_global_frame(index_json_path: str) -> GlobalFrame:
    with open(index_json_path, "r") as f:
        d = json.load(f)

    # plane
    if "plane_n_d" in d and isinstance(d["plane_n_d"], list):
        N_list, d_val = d["plane_n_d"]
    else:
        N_list = _get(d, "plane_n", default=[0,0,1])
        d_val  = _get(d, "plane_d", default=0.0)
    N = np.asarray(N_list, dtype=np.float32)
    N = N / (np.linalg.norm(N) + 1e-12)
    d0 = float(d_val)

    # frame
    R = np.asarray(_get(d, "R_cols_world"), dtype=np.float32)
    if R.shape != (3,3):
        R = np.eye(3, dtype=np.float32)
    o = np.asarray(_get(d, "origin_world", default=[0,0,0]), dtype=np.float32)

    # geometry
    mpp = float(_get(d, "mpp", "target_mpp", default=1.0))
    # <- THIS is the compatibility fix:
    tile_px = int(_get(d, "tile_px", "grid_size_px", default=1024))

    grid = _get(d, "grid", default=None) or {}
    nx = int(_get(d, "nx", default=_get(grid, "Nu", default=1)))
    ny = int(_get(d, "ny", default=_get(grid, "Nv", default=1)))

    # raster origin in plane
    u0v0 = _get(d, "plane_origin_uv", default=None)
    if isinstance(u0v0, (list, tuple)) and len(u0v0) == 2:
        u0, v0 = float(u0v0[0]), float(u0v0[1])
    else:
        bbox = _get(d, "bbox_global", default=[0.0, 0.0, nx*tile_px*mpp, ny*tile_px*mpp])
        u0, v0 = float(bbox[0]), float(bbox[1])

    clip_lo = float(_get(d, "clip_lo", default=1.0))
    clip_hi = float(_get(d, "clip_hi", default=99.0))

    return GlobalFrame(N=N, d=d0, R=R, o=o,
                       mpp=mpp, tile_px=tile_px, nx=nx, ny=ny,
                       u0=u0, v0=v0, clip_lo=clip_lo, clip_hi=clip_hi)

# --- helpers ---

def world_to_plane(P_world: np.ndarray, gf: GlobalFrame) -> np.ndarray:
    P = np.asarray(P_world, dtype=np.float32)
    uvw = (P - gf.o[None, :]) @ gf.R
    return uvw[:, :2].astype(np.float32)

def plane_to_tile_idx(u: float, v: float, gf: GlobalFrame) -> Tuple[int,int]:
    up = (u - gf.u0) / gf.mpp
    vp = (v - gf.v0) / gf.mpp
    tx = int(math.floor(up / gf.tile_px))
    ty = int(math.floor(vp / gf.tile_px))
    return tx, ty

def tile_bbox(tx: int, ty: int, gf: GlobalFrame) -> Tuple[float,float,float,float]:
    x0 = gf.u0 + tx * gf.tile_px * gf.mpp
    y0 = gf.v0 + ty * gf.tile_px * gf.mpp
    x1 = x0 + gf.tile_px * gf.mpp
    y1 = y0 + gf.tile_px * gf.mpp
    return float(x0), float(y0), float(x1), float(y1)
