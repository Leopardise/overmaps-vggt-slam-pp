from __future__ import annotations
import os, json, math, tempfile
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import cv2


# ---------------------------
# Atomic writers
# ---------------------------

def _write_json_atomic(path: str, obj: dict) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _save_npy_atomic(path: str, arr: np.ndarray) -> None:
    d = os.path.dirname(path)
    Path(d).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".tmp_", suffix=".npy", dir=d, delete=False) as tf:
        np.save(tf, arr)
        tf.flush(); os.fsync(tf.fileno())
        tmp = tf.name
    os.replace(tmp, path)


# ---------------------------
# Map point collection & plane fit
# ---------------------------

def _collect_all_points_world(graph_map, stride: int = 1) -> np.ndarray:
    for name in ["get_all_points_in_world_frame", "get_all_points_world", "get_points_world"]:
        fn = getattr(graph_map, name, None)
        if callable(fn):
            try:
                P = fn(stride=stride) if "stride" in fn.__code__.co_varnames else fn()
                P = np.asarray(P, dtype=np.float32).reshape(-1, 3)
                if P.size:
                    return P
            except Exception:
                pass
    # fallback: aggregate submaps
    submaps = []
    get_sms = getattr(graph_map, "get_submaps", None)
    if callable(get_sms):
        try:
            raw = get_sms()
            if isinstance(raw, dict): raw = raw.values()
            for sm in list(raw):
                for n in ["get_points_in_world_frame", "get_points_world", "get_points"]:
                    fn = getattr(sm, n, None)
                    if callable(fn):
                        try:
                            Q = fn()
                            Q = np.asarray(Q, dtype=np.float32).reshape(-1, 3)
                            if Q.size:
                                submaps.append(Q)
                                break
                        except Exception:
                            pass
        except Exception:
            pass
    if submaps:
        return np.concatenate(submaps, axis=0).astype(np.float32)
    return np.zeros((0, 3), np.float32)

def _fit_plane(points: np.ndarray, iters: int = 600, thresh: float = 0.02) -> Tuple[np.ndarray, float]:
    if points.shape[0] < 3:
        return np.array([0, 0, 1], np.float32), 0.0

    P = np.asarray(points, dtype=np.float32)
    # strip non-finite
    mask = np.isfinite(P).all(axis=1)
    if mask.sum() < 3:
        return np.array([0, 0, 1], np.float32), 0.0
    P = P[mask]

    N = P.shape[0]
    if N >= 1000:
        best_in = None; best_ct = 0
        rng = np.random.default_rng(0xC0FFEE)
        for _ in range(iters):
            idx = rng.choice(N, 3, replace=False)
            A = P[idx]
            v1 = A[1] - A[0]; v2 = A[2] - A[0]
            n = np.cross(v1, v2); n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            d = -float(n @ A[0])
            dist = np.abs(P @ n + d)
            mask_in = dist < thresh
            ct = int(mask_in.sum())
            if ct > best_ct:
                best_ct, best_in = ct, mask_in

        if best_in is not None and best_ct >= 100:
            P2 = P[best_in]
            c = P2.mean(0)
            try:
                U, S, Vt = np.linalg.svd((P2 - c).astype(np.float64), full_matrices=False)
                n = Vt[-1]; n /= (np.linalg.norm(n) + 1e-12)
            except np.linalg.LinAlgError:
                C = np.cov((P2 - c).T)
                evals, evecs = np.linalg.eigh(C)
                n = evecs[:, np.argmin(evals)]
                n /= (np.linalg.norm(n) + 1e-12)
            d = -float(n @ c)
            return n.astype(np.float32), d

    # fallback least-squares plane
    c = P.mean(0)
    try:
        U, S, Vt = np.linalg.svd((P - c).astype(np.float64), full_matrices=False)
        n = Vt[-1]
    except np.linalg.LinAlgError:
        C = np.cov((P - c).T)
        evals, evecs = np.linalg.eigh(C)
        n = evecs[:, np.argmin(evals)]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(n @ c)
    return n.astype(np.float32), d

def _plane_frame(n: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an orthonormal frame aligned to the plane with normal n that best fits `points`.
    Robust to outliers, extreme dynamic ranges, and huge N.

    Strategy:
      * Center & precondition points (remove non-finite, clip heavy tails, scale).
      * Project to plane → Pz.
      * Try SVD on a capped random subset (fast, stable).
      * If SVD fails, fall back to eig on covariance(Pz).
      * If that fails, use a deterministic default basis.

    Returns:
      R: 3x3 rotation with columns [x, y, z], where z ≡ n/||n||
      o: 3-vector plane origin (mean of valid points)
    """
    # --- guard & normalize normal ---
    n = np.asarray(n, dtype=np.float32)
    n = n / (np.linalg.norm(n) + 1e-12)
    P = np.asarray(points, dtype=np.float32)

    # --- remove non-finite ---
    mask = np.isfinite(P).all(axis=1)
    if not np.any(mask):
        # Fallback: identity-orientation frame at origin
        R = np.eye(3, dtype=np.float32)
        return R, np.zeros(3, dtype=np.float32)
    P = P[mask]

    # --- origin at mean (more stable than median for dense point clouds) ---
    o = P.mean(axis=0).astype(np.float32)

    # --- project to plane (remove component along n) ---
    P0 = P - o[None, :]
    Pz = P0 - np.outer(P0 @ n, n)  # shape (M,3)

    # --- heavy-tail clipping (robust preconditioning) ---
    # Clip by radial distance within the plane to 99.9 percentile to avoid huge leverage
    r = np.linalg.norm(Pz, axis=1)
    if r.size >= 10:
        r_thr = np.percentile(r, 99.9)
        keep = r <= (r_thr + 1e-9)
        if keep.sum() >= 3:
            Pz = Pz[keep]
            r = r[keep]

    # --- scale to unit spread to help SVD convergence ---
    s = np.median(r) if r.size else 1.0
    s = float(s) if (np.isfinite(s) and s > 0) else 1.0
    Pz_scaled = (Pz / s).astype(np.float64, copy=False)  # SVD likes float64

    # --- cap sample size for SVD ---
    MAX_SVD_SAMPLES = 200000  # plenty for stable PCA in 2D; adjust if needed
    M = Pz_scaled.shape[0]
    if M > MAX_SVD_SAMPLES:
        rng = np.random.default_rng(12345)
        idx = rng.choice(M, size=MAX_SVD_SAMPLES, replace=False)
        Pz_scaled_s = Pz_scaled[idx]
    else:
        Pz_scaled_s = Pz_scaled

    # --- try SVD, then covariance-eig as fallback ---
    x = None
    try:
        # SVD on centered data (already centered by construction)
        # We want the first principal direction in the plane.
        U, S, Vt = np.linalg.svd(Pz_scaled_s, full_matrices=False)
        # Vt rows are principal axes; the one with largest singular value is along max variance
        x = Vt[0].astype(np.float64)
    except np.linalg.LinAlgError:
        try:
            # Covariance-based eigendecomposition (more forgiving for some LAPACK builds)
            C = (Pz_scaled_s.T @ Pz_scaled_s) / max(1, Pz_scaled_s.shape[0] - 1)
            evals, evecs = np.linalg.eigh(C)  # ascending order
            x = evecs[:, np.argmax(evals)].astype(np.float64)
        except Exception:
            x = None

    # --- normalize & orthogonalize with n; final fallback if needed ---
    if x is None or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-12:
        # Deterministic default basis orthogonal to n
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(a @ n)) > 0.9:
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x = a - (a @ n) * n
        x /= (np.linalg.norm(x) + 1e-12)

    # ensure orthonormal (x, y, z)
    z = n.astype(np.float64)
    x = (x - (x @ z) * z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x);     y /= (np.linalg.norm(y) + 1e-12)

    R = np.stack([x, y, z], axis=1).astype(np.float32)
    return R, o

def perturb_plane(n: np.ndarray, d: float,
                  rot_deg: float = 1.0,
                  trans_m: float = 0.05,
                  seed: int = 0):
    """
    Small, controlled perturbation of plane parameters.
    """
    rng = np.random.default_rng(seed)

    # random small rotation
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    theta = np.deg2rad(rot_deg) * rng.uniform(-1, 1)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    n_p = (R @ n)
    n_p /= np.linalg.norm(n_p) + 1e-12

    # small translation
    d_p = d + rng.uniform(-trans_m, trans_m)

    return n_p.astype(np.float32), float(d_p)


def _world_to_plane_xy(points: np.ndarray, R: np.ndarray, o: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = points - o[None, :]
    uvw = p @ R
    return uvw[:, :2].astype(np.float32), uvw[:, 2].astype(np.float32)


# ---------------------------
# Rasterization & grid
# ---------------------------

def _calc_bbox(uv: np.ndarray, radius_keep_pct: float = 100.0) -> Tuple[float, float, float, float]:
    if uv.shape[0] == 0:
        return 0, 0, 1, 1
    c = uv.mean(0)
    r = np.sqrt(((uv - c) ** 2).sum(1))
    if 0 < radius_keep_pct < 100.0:
        thr = np.percentile(r, radius_keep_pct)
        uv = uv[r <= thr]
    u0, v0 = uv.min(0); u1, v1 = uv.max(0)
    return float(u0), float(v0), float(u1), float(v1)

def _mpp_from_target(span_long: float, target_px_long: int) -> float:
    return float(span_long) / float(max(1, int(target_px_long)))

def _tile_grid(u0, v0, u1, v1, mpp: float, tile_px: int) -> Tuple[int, int, float, float]:
    width = u1 - u0; height = v1 - v0
    Wpx = int(math.ceil(width / mpp))
    Hpx = int(math.ceil(height / mpp))
    Nu = int(math.ceil(Wpx / tile_px))
    Nv = int(math.ceil(Hpx / tile_px))
    return Nu, Nv, float(u0), float(v0)

def _reducer_factory(name: str, tau: float):
    n = name.lower()
    if n == "mean":
        return lambda zs: float(zs.mean()) if zs.size else np.nan
    if n == "max":
        return lambda zs: float(zs.max()) if zs.size else np.nan
    if n == "softmax":
        def f(zs: np.ndarray) -> float:
            if zs.size == 0: return np.nan
            t = max(1e-6, float(tau))
            a = (zs / t).astype(np.float64); a -= a.max()
            w = np.exp(a)
            return float((w * zs).sum() / (w.sum() + 1e-12))
        return f
    raise ValueError(f"Unknown reducer: {name}")

def _splat_tile(uv: np.ndarray, h: np.ndarray,
                u0: float, v0: float, mpp: float, tile_px: int,
                Iu: int, Iv: int, reducer) -> np.ndarray:
    up = (uv[:, 0] - u0) / mpp
    vp = (uv[:, 1] - v0) / mpp
    x0 = Iu * tile_px; y0 = Iv * tile_px
    x1 = x0 + tile_px; y1 = y0 + tile_px

    mask = (up >= x0) & (up < x1) & (vp >= y0) & (vp < y1)
    if not np.any(mask):
        return np.full((tile_px, tile_px), np.nan, np.float32)

    up_t = up[mask] - x0
    vp_t = vp[mask] - y0
    h_t = h[mask].astype(np.float32)

    ix = np.clip(up_t.round().astype(np.int32), 0, tile_px - 1)
    iy = np.clip(vp_t.round().astype(np.int32), 0, tile_px - 1)

    tile = np.full((tile_px, tile_px), np.nan, np.float32)
    buckets: Dict[int, List[float]] = {}
    flat_idx = (iy * tile_px + ix).astype(np.int64)
    for fi, zi in zip(flat_idx.tolist(), h_t.tolist()):
        buckets.setdefault(fi, []).append(zi)
    for fi, lst in buckets.items():
        y = fi // tile_px; x = fi % tile_px
        tile[y, x] = reducer(np.asarray(lst, np.float32))
    return tile


# === PUBLIC: used by submap_chipper ===
def _rasterize_tile(
    XY_uv: np.ndarray,          # (N,2)
    Z_h:  np.ndarray,           # (N,)
    tile_bbox_m: tuple,         # (x0,y0,x1,y1)
    mpp: float,                 # meters per pixel
    kernel_px: float,           # (kept for API compatibility)
    reducer: str,               # "softmax" | "mean" | "max"
    softmax_tau: float,         # tau if reducer == softmax
    tile_px_fixed: Optional[int] = None,  # force output size to match global tile_px
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = map(float, tile_bbox_m)

    if tile_px_fixed is not None:
        tile_px = int(tile_px_fixed)
    else:
        tile_px_x = int(round((x1 - x0) / mpp))
        tile_px_y = int(round((y1 - y0) / mpp))
        tile_px = max(1, min(tile_px_x, tile_px_y))  # assume square tiles

    reducer_fn = _reducer_factory(reducer, softmax_tau)

    up = (XY_uv[:, 0] - x0) / mpp
    vp = (XY_uv[:, 1] - y0) / mpp
    mask_pts = (up >= 0) & (up < tile_px) & (vp >= 0) & (vp < tile_px)
    if not np.any(mask_pts):
        dem = np.full((tile_px, tile_px), np.nan, np.float32)
        occ = np.zeros((tile_px, tile_px), np.bool_)
        return dem, occ

    up_t = up[mask_pts]; vp_t = vp[mask_pts]; h_t = Z_h[mask_pts].astype(np.float32)
    ix = np.clip(np.round(up_t).astype(np.int32), 0, tile_px - 1)
    iy = np.clip(np.round(vp_t).astype(np.int32), 0, tile_px - 1)

    dem = np.full((tile_px, tile_px), np.nan, np.float32)
    occ = np.zeros((tile_px, tile_px), np.bool_)
    buckets: Dict[int, List[float]] = {}
    flat_idx = (iy * tile_px + ix).astype(np.int64)
    for fi, zi in zip(flat_idx.tolist(), h_t.tolist()):
        buckets.setdefault(fi, []).append(zi)
    for fi, lst in buckets.items():
        y = fi // tile_px; x = fi % tile_px
        dem[y, x] = reducer_fn(np.asarray(lst, np.float32))
        occ[y, x] = True
    return dem, occ
# === end PUBLIC ===


# ---------------------------
# Visualization helpers (white bg)
# ---------------------------

def _edge_mask(dem: np.ndarray, mpp: float, strength: float) -> np.ndarray:
    if not np.isfinite(dem).any():
        return np.ones_like(dem, np.float32)
    z = dem.copy()
    z[~np.isfinite(z)] = np.nanmean(z[np.isfinite(z)]) if np.isfinite(z).any() else 0.0
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    g = g / (np.nanpercentile(g, 99.0) + 1e-9)
    g = np.clip(g, 0, 1)
    return (1.0 - strength * g).astype(np.float32)

def _hillshade_mask(dem: np.ndarray, mpp: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    if not np.isfinite(dem).any():
        return np.ones_like(dem, np.float32)
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    zx = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3) / (mpp + 1e-9)
    zy = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3) / (mpp + 1e-9)
    slope = np.arctan(np.hypot(zx, zy))
    aspect = np.arctan2(-zy, zx)
    hs = (np.sin(alt) * np.cos(slope) +
          np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    hs = (hs - hs.min()) / (hs.max() - hs.min() + 1e-9)
    return hs.astype(np.float32)

def _grayscale_viz_from_dem(
    dem: np.ndarray,
    mpp: float,
    edge_strength: float,
    shade_strength: float,
    dark_level: float,
    unsharp_radius_px: float,
    unsharp_amount: float,
    clahe_clip: float,
    clahe_grid: int,
    *,
    lo_hi: tuple | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    DEM → BGR, consistent global scaling + optional gamma; WHITE background where NaN.
    """
    g = dem.astype(np.float32).copy()
    mask = np.isfinite(g)

    if mask.any():
        if lo_hi is None:
            lo, hi = np.percentile(g[mask], (1.0, 99.0))
        else:
            lo, hi = float(lo_hi[0]), float(lo_hi[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = g[mask].min(), g[mask].max() + 1e-6
        g = (g - lo) / (hi - lo)
        g = np.clip(g, 0, 1)
    else:
        g = np.zeros_like(g, np.float32)

    edge = _edge_mask(dem, mpp, edge_strength)
    shade = _hillshade_mask(dem, mpp)
    mix = (1.0 - shade_strength) + shade_strength * shade

    img = np.clip(g * edge * mix, 0, 1)
    if dark_level > 0:
        img *= (1.0 - 0.55 * dark_level)

    if gamma != 1.0:
        img = np.power(img, max(1e-3, gamma))

    # Avoid warnings/corruption if img contains NaN/inf
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    gray = (img * 255.0).astype(np.uint8)

    # Unsharp & CLAHE
    if unsharp_amount > 0 and unsharp_radius_px > 0:
        k = int(max(1, round(unsharp_radius_px))) * 2 + 1
        blur = cv2.GaussianBlur(gray, (k, k), unsharp_radius_px)
        gray = np.clip(gray + unsharp_amount * (gray - blur), 0, 255).astype(np.uint8)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(clahe_grid, clahe_grid))
        gray = clahe.apply(gray)

    # WHITE background where NaN
    gray[~mask] = 255
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _colorize_gray_white_bg(gray_bgr: np.ndarray, cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    gray = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)
    color = cv2.applyColorMap(gray, cmap)
    bg = (gray == 255)
    if bg.any():
        color[bg] = (255, 255, 255)
    return color


# ---------------------------
# Quicklook mosaic
# ---------------------------

def _safe_quicklook_layout(Nu: int, Nv: int, tile_px: int, quick_max_long: int = 4096) -> Tuple[int, int, int]:
    scaled = max(1, int(min(tile_px, max(1, quick_max_long // max(Nu, Nv)))))
    W = Nu * scaled; H = Nv * scaled
    return scaled, W, H


# ---------------------------
# RViz2 publisher (optional)
# ---------------------------

def _publish_rviz2_image(img_bgr: np.ndarray, topic: str = "/heightmap/image"):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
    except Exception:
        return  # ROS 2 not available; skip

    class ImagePub(Node):
        def __init__(self, image):
            super().__init__("global_dem_quicklook_pub")
            self.pub = self.create_publisher(Image, topic, 1)
            self.br = CvBridge()
            self.msg = self.br.cv2_to_imgmsg(image, encoding="bgr8")
            for _ in range(5):
                self.pub.publish(self.msg)
                rclpy.spin_once(self, timeout_sec=0.1)

    try:
        rclpy.init()
        ImagePub(img_bgr)
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass

def height_from_gray(gray: np.ndarray, viz_lo: float, viz_hi: float) -> np.ndarray:
    """
    Recover metric height (meters) from grayscale visualization.
    """
    h = viz_lo + (gray.astype(np.float32) / 255.0) * (viz_hi - viz_lo)
    return h

def build_grayscale_height_lut(viz_lo: float, viz_hi: float) -> np.ndarray:
    """
    256-entry LUT:
    gray value [0..255] -> metric height (meters)
    """
    gray = np.arange(256, dtype=np.float32)
    return viz_lo + (gray / 255.0) * (viz_hi - viz_lo)


# ---------------------------
# Entry point
# ---------------------------

def render_global_dem_tiled(
    graph_map,
    out_dir: str,
    radius_keep_pct: float,
    clip_lo: float, clip_hi: float,
    target_px_long: int,
    tile_px: int,
    kernel_px: float,
    reducer: str,
    softmax_tau: float,
    cycle_meters: float,
    edge_strength: float,
    shade_strength: float,
    dark_level: float,
    unsharp_radius_px: float,
    unsharp_amount: float,
    clahe_clip: float,
    clahe_grid: int,
    stride: int = 1,
    quick_max_long: int = 4096,
    publish_rviz: bool = True,
) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tiles_dir = out / "tiles"; tiles_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect points & plane
    P = _collect_all_points_world(graph_map, stride=stride).astype(np.float32)
    if P.shape[0] < 100:
        print("[global_dem_tiled] Not enough points; skipping render.")
        return
    n, d = _fit_plane(P)
    n_p, d_p = perturb_plane(n, d, rot_deg=5.0, trans_m=2.0)
    R_p, o_p = _plane_frame(n_p, P)

    # Shift origin so plane offset is respected
    # plane: n·x + d = 0  ⇒ shift origin by -d * n
    o_p = o_p - d_p * n_p

    uv_p, h_p = _world_to_plane_xy(P, R_p, o_p)


    R, o = _plane_frame(n, P)
    uv, h = _world_to_plane_xy(P, R, o)

    # 2) Clip heights & bbox
    if np.isfinite(h).any():
        lo = np.percentile(h, clip_lo) if clip_lo > 0 else np.min(h)
        hi = np.percentile(h, clip_hi) if clip_hi < 100 else np.max(h)
        h = np.clip(h, lo, hi)
        h_p = np.clip(h_p, lo, hi)


    u0, v0, u1, v1 = _calc_bbox(uv, radius_keep_pct=radius_keep_pct)
    span_long = max(u1 - u0, v1 - v0)
    if span_long <= 0:
        print("[global_dem_tiled] Degenerate span; skipping.")
        return

    mpp = _mpp_from_target(span_long, target_px_long)
    Nu, Nv, u_start, v_start = _tile_grid(u0, v0, u1, v1, mpp, tile_px)
    reducer_fn = _reducer_factory(reducer, softmax_tau)

    # Global absolute grayscale for consistent tiles & submap chips
    finite_h = np.isfinite(h)
    if finite_h.any():
        viz_lo = float(np.percentile(h[finite_h], 0.5))
        viz_hi = float(np.percentile(h[finite_h], 99.5))
        if (not np.isfinite(viz_lo)) or (not np.isfinite(viz_hi)) or (viz_hi <= viz_lo):
            viz_lo = float(np.nanmin(h)); viz_hi = float(np.nanmax(h)) + 1e-6
    else:
        viz_lo, viz_hi = 0.0, 1.0

    # 3) Per-tile DEM + grayscale PNG (white background)
    tile_paths_png, tile_paths_npy = [], []
    rmse_accum = []

    for Iv in range(Nv):
        for Iu in range(Nu):
            dem = _splat_tile(uv, h, u_start, v_start, mpp, tile_px, Iu, Iv, reducer_fn)
            dem_p = _splat_tile(uv_p, h_p, u_start, v_start, mpp, tile_px, Iu, Iv, reducer_fn)
            mask = np.isfinite(dem) & np.isfinite(dem_p)
            if mask.any():
                rmse_accum.append(np.mean((dem[mask] - dem_p[mask])**2))

            tid = Iv * Nu + Iu

            npy_path = tiles_dir / f"tile_{tid:05d}.npy"
            _save_npy_atomic(str(npy_path), dem.astype(np.float32))
            tile_paths_npy.append(str(npy_path.relative_to(out)))

            png_path = tiles_dir / f"tile_{tid:05d}.png"
            g = _grayscale_viz_from_dem(
                dem=dem, mpp=mpp,
                edge_strength=edge_strength,
                shade_strength=shade_strength,
                dark_level=dark_level,
                unsharp_radius_px=unsharp_radius_px,
                unsharp_amount=unsharp_amount,
                clahe_clip=clahe_clip,
                clahe_grid=clahe_grid,
                lo_hi=(viz_lo, viz_hi),
                gamma=0.85,  # slight midtone lift
            )
            cv2.imwrite(str(png_path), g)
            tile_paths_png.append(str(png_path.relative_to(out)))

    rmse = float(np.sqrt(np.mean(rmse_accum))) if rmse_accum else float("nan")
    
    # 4) Quicklook mosaics
    scaled_tile_px, Wq, Hq = _safe_quicklook_layout(Nu, Nv, tile_px, quick_max_long=quick_max_long)
    mosaic = np.full((Hq, Wq, 3), 255, np.uint8)  # WHITE background
    # === COLOR ↔ HEIGHT CONTRACT (EXPLICIT) ===
    height_lut = build_grayscale_height_lut(viz_lo, viz_hi)
    np.save(out / "gray_to_height_lut.npy", height_lut)

    t = 0
    for Iv in range(Nv):
        for Iu in range(Nu):
            png_path = out / tile_paths_png[t]; t += 1
            img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
            if img is None: continue
            if img.shape[0] != scaled_tile_px or img.shape[1] != scaled_tile_px:
                img = cv2.resize(img, (scaled_tile_px, scaled_tile_px), interpolation=cv2.INTER_AREA)
            y0 = Iv * scaled_tile_px; x0 = Iu * scaled_tile_px
            mosaic[y0:y0 + scaled_tile_px, x0:x0 + scaled_tile_px] = img

    mosaic_gray_path = out / "mosaic_quicklook.png"
    cv2.imwrite(str(mosaic_gray_path), mosaic)

    colorized = _colorize_gray_white_bg(mosaic, cmap=cv2.COLORMAP_VIRIDIS)
    mosaic_color_path = out / "mosaic_quicklook_color.png"
    cv2.imwrite(str(mosaic_color_path), colorized)

    # 5) Index JSON (+ viz scale)
    index = {
        "bbox_global": [u0, v0, u1, v1],
        "plane_origin_uv": [u_start, v_start],
        "origin_world": o.tolist(),
        "R_cols_world": R.tolist(),           # columns are [x y z]
        "plane_n_d": [n.tolist(), d],
        "plane_n": n.tolist(),
        "plane_d": d,
        "mpp": float(mpp),
        "target_px_long": int(target_px_long),
        "target_mpp": float(mpp),
        "tile_px": int(tile_px),
        "grid": {"Nu": int(Nu), "Nv": int(Nv)},
        "nx": int(Nu), "ny": int(Nv),         # convenience
        "tiles_png": tile_paths_png,
        "tiles_npy": tile_paths_npy,
        "mosaic_quicklook": "mosaic_quicklook.png",
        "mosaic_quicklook_color": "mosaic_quicklook_color.png",
        # expose the EXACT absolute scaling used for all visuals:
        "viz_lo": float(viz_lo),
        "viz_hi": float(viz_hi),
        # legacy percentile hints (if other tools read them)
        "clip_lo": float(0.5),
        "clip_hi": float(99.5),
        "visual_params": {
            "edge_strength": float(edge_strength),
            "shade_strength": float(shade_strength),
            "dark_level": float(dark_level),
            "unsharp_radius_px": float(unsharp_radius_px),
            "unsharp_amount": float(unsharp_amount),
            "clahe_clip": float(clahe_clip),
            "clahe_grid": int(clahe_grid),
            "reducer": reducer,
            "softmax_tau": float(softmax_tau),
            "radius_keep_pct": float(radius_keep_pct),
        },
        "color_mapping": {
            "space": "grayscale_then_colormap",
            "grayscale_range": [0, 255],
            "height_range_m": [viz_lo, viz_hi],
            "gray_to_height_lut": "gray_to_height_lut.npy",
            "colormap": "viridis",
            "note": "Viridis is applied AFTER grayscale; heights are encoded in grayscale pre-color."
        },
        "plane_perturbation_test": {
            "enabled": True,
            "rotation_deg": 1.0,
            "translation_m": 0.05,
            "rmse_m": rmse,
            "note": "RMSE between DEMs generated from original and perturbed plane fits"
        },



    }
    _write_json_atomic(str(out / "index.json"), index)

    # 6) Optional RViz2 publish
    if publish_rviz:
        img = cv2.imread(str(mosaic_color_path), cv2.IMREAD_COLOR)
        if img is not None:
            _publish_rviz2_image(img)
            
    # valid = np.isfinite(dem) & np.isfinite(dem_p)
    # rmse = np.sqrt(np.mean((dem[valid] - dem_p[valid])**2))
    # diff = np.abs(dem - dem_p)
    # diff_img = colorize_dem_metric(diff, 0.0, np.percentile(diff, 99))
    # cv2.imwrite("dem_perturb_diff.png", diff_img)


    print(f"[global_dem_tiled] DONE → {out}")
    print(f"  tiles:        {len(tile_paths_png)}  ({Nu} × {Nv}, {tile_px}px each)")
    print(f"  quicklook:    {mosaic_gray_path.name}  ({Wq}×{Hq})")
    print(f"  quicklook(c): {mosaic_color_path.name}")
    print(f"  index:        index.json")
