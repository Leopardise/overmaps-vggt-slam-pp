from __future__ import annotations
import os, json, glob, math, argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import cv2

# ----- import your project helpers -----
from vggt_slam.covis.frame_io import (
    load_global_frame, world_to_plane, plane_to_tile_idx, tile_bbox, GlobalFrame
)
from vggt_slam.global_dem_tiled import _rasterize_tile  # uses same binning as renderer


# ========= Grayscale helpers (white background) =========

def _u8_from_dem_white_bg(
    dem: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Map DEM float32 to uint8 0..255, with 255 as WHITE background for NaNs.
    'lo'/'hi' are absolute global thresholds (meters along plane normal).
    """
    u8 = np.full(dem.shape, 255, np.uint8)  # white by default
    mask = np.isfinite(dem)
    if mask.any():
        g = (np.clip(dem[mask], lo, hi) - lo) / (hi - lo + 1e-12)
        g = np.clip(g, 0.0, 1.0)
        u8[mask] = (g * 255.0 + 0.5).astype(np.uint8)
    return u8


def _png_from_dem_white_bg(
    dem: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """
    DEM -> BGR PNG array; grayscale with WHITE where NaN.
    """
    u8 = _u8_from_dem_white_bg(dem, lo, hi)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


# ========= Global lo/hi from global tiles (fast sampling) =========

def _approx_global_lohi_from_tiles(
    tiles_glob: str,
    clip_lo_pct: float,
    clip_hi_pct: float,
    sample_stride: int = 32,
    max_samples: int = 5_000_000,
) -> Tuple[float, float]:
    """
    Quickly approximate global lo/hi using a stride sampler over existing
    global tile .npy DEMs. Works out-of-core.

    Returns absolute (lo, hi) in meters (signed height along plane normal).
    """
    samples = []
    for p in sorted(glob.glob(tiles_glob)):
        try:
            z = np.load(p).astype(np.float32, copy=False)
        except Exception:
            continue
        if z.size == 0:
            continue
        zf = z[np.isfinite(z)]
        if zf.size == 0:
            continue
        # stride-sample
        if sample_stride > 1:
            zf = zf[::sample_stride]
        samples.append(zf)
        # memory cap
        if sum(arr.size for arr in samples) > max_samples:
            break

    if not samples:
        # fallback if no usable tiles found
        return 0.0, 1.0

    allz = np.concatenate(samples, axis=0)
    lo = float(np.percentile(allz, clip_lo_pct))
    hi = float(np.percentile(allz, clip_hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(allz)), float(np.nanmax(allz)) + 1e-6
    return lo, hi


# ========= Core chipping =========

def _signed_height_world(P_world: np.ndarray, gf: GlobalFrame) -> np.ndarray:
    """
    Signed height along the global plane normal: z = N·p + d
    """
    return (P_world @ gf.N.astype(np.float64)) + float(gf.d)


def _overlapping_tiles_for_bbox(
    sx0: float, sy0: float, sx1: float, sy1: float, gf: GlobalFrame
) -> Tuple[int, int, int, int]:
    tx0, ty0 = plane_to_tile_idx(sx0, sy0, gf)
    tx1, ty1 = plane_to_tile_idx(sx1, sy1, gf)
    tx0 = max(0, min(gf.nx - 1, tx0)); tx1 = max(0, min(gf.nx - 1, tx1))
    ty0 = max(0, min(gf.ny - 1, ty0)); ty1 = max(0, min(gf.ny - 1, ty1))
    return tx0, ty0, tx1, ty1


def _chip_one_submap(
    root: str,
    sm_dir: str,
    gf: GlobalFrame,
    lo_global: float,
    hi_global: float,
    reducer: str,
    softmax_tau: float,
    kernel_px: float,
    overwrite: bool,
) -> Dict:
    submap_id = os.path.basename(sm_dir.rstrip("/"))
    pts_file  = os.path.join(sm_dir, "points_world.npy")
    if not os.path.isfile(pts_file):
        return {"submap_id": submap_id, "overlapped_tile_ids": [], "chips_dir": ""}

    try:
        P_world = np.load(pts_file).astype(np.float32, copy=False)
    except Exception:
        return {"submap_id": submap_id, "overlapped_tile_ids": [], "chips_dir": ""}

    if P_world.size == 0:
        return {"submap_id": submap_id, "overlapped_tile_ids": [], "chips_dir": ""}

    # Project to frozen global plane coords
    XY = world_to_plane(P_world, gf)  # (N,2)
    Z  = _signed_height_world(P_world, gf).astype(np.float32)

    sub_dir   = os.path.join(root, "submaps", submap_id)
    chips_dir = os.path.join(sub_dir, "chips")
    Path(chips_dir).mkdir(parents=True, exist_ok=True)

    # submap bbox (meters in plane)
    sx0, sy0 = XY.min(axis=0).tolist()
    sx1, sy1 = XY.max(axis=0).tolist()

    tx0, ty0, tx1, ty1 = _overlapping_tiles_for_bbox(sx0, sy0, sx1, sy1, gf)

    # persist meta (for debugging/inspection)
    with open(os.path.join(sub_dir, "meta.json"), "w") as fmeta:
        json.dump({
            "sm_id": submap_id,
            "plane_bbox": [float(sx0), float(sy0), float(sx1), float(sy1)],
            "mpp": float(gf.mpp),
            "tile_px": int(gf.tile_px),
            "frame_index_json": os.path.abspath(os.path.join(root, "index.json")),
            "reducer": reducer,
            "softmax_tau": float(softmax_tau),
            "kernel_px": float(kernel_px),
            "global_lo": float(lo_global),
            "global_hi": float(hi_global),
        }, fmeta, indent=2)

    overlapped_tiles: List[int] = []
    wrote_any = False

    for ty in range(min(ty0, ty1), max(ty0, ty1) + 1):
        for tx in range(min(tx0, tx1), max(tx0, tx1) + 1):
            # tile bbox in plane
            tb = tile_bbox(tx, ty, gf)  # (u0,v0,u1,v1)
            u0, v0, u1, v1 = tb

            inside = (XY[:, 0] >= u0) & (XY[:, 0] < u1) & (XY[:, 1] >= v0) & (XY[:, 1] < v1)
            if not np.any(inside):
                continue

            tid = ty * gf.nx + tx
            npy_path = os.path.join(chips_dir, f"{tid:05d}.npy")
            png_path = os.path.join(chips_dir, f"{tid:05d}.png")

            if (not overwrite) and os.path.exists(npy_path) and os.path.exists(png_path):
                overlapped_tiles.append(int(tid))
                wrote_any = True
                continue

            dem_chip, occ_chip = _rasterize_tile(
                XY[inside], Z[inside], tb,
                gf.mpp, kernel_px, reducer, softmax_tau
            )

            # save chip DEM
            np.save(npy_path, dem_chip.astype(np.float32))

            # preview PNG (white bg) using *global* lo/hi
            png = _png_from_dem_white_bg(dem_chip, lo_global, hi_global)
            # Make the truly-empty pixels white too (occ=false or NaN)
            mask = np.isfinite(dem_chip)
            if mask.any():
                dil = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
                png[~dil] = (255, 255, 255)
            cv2.imwrite(png_path, png)

            overlapped_tiles.append(int(tid))
            wrote_any = True

    # If nothing overlapped, also emit a local fallback chip in the same style/scale.
    if not wrote_any:
        # compute a tight local bbox snapped to one tile in plane grid units
        # use the submap bbox; snap to tile size in meters
        tile_m = gf.tile_px * gf.mpp
        # snap lower-left
        u0 = math.floor(sx0 / tile_m) * tile_m
        v0 = math.floor(sy0 / tile_m) * tile_m
        u1 = u0 + tile_m
        v1 = v0 + tile_m
        tb = (u0, v0, u1, v1)

        dem_chip, occ_chip = _rasterize_tile(XY, Z, tb, gf.mpp, kernel_px, reducer, softmax_tau)

        base = os.path.join(chips_dir, f"local_{submap_id}")
        np.save(base + ".npy", dem_chip.astype(np.float32))
        png = _png_from_dem_white_bg(dem_chip, lo_global, hi_global)
        mask = np.isfinite(dem_chip)
        if mask.any():
            dil = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
            png[~dil] = (255, 255, 255)
        else:
            png[:] = (255, 255, 255)
        cv2.imwrite(base + ".png", png)

    # tiles.json
    with open(os.path.join(sub_dir, "tiles.json"), "w") as f:
        json.dump({"overlapped_tile_ids": overlapped_tiles}, f, indent=2)

    return {
        "submap_id": submap_id,
        "overlapped_tile_ids": overlapped_tiles,
        "chips_dir": chips_dir,
    }


# ========= CLI =========

def main():
    ap = argparse.ArgumentParser("Chip saved submap point clouds into DEM chips (global-compatible).")
    ap.add_argument("--root", required=True, help="Run root, e.g. outputs/00 (must contain index.json & tiles/)")
    ap.add_argument("--submap", default="", help="Optional: only chip this submap id (e.g., sm_00012).")
    ap.add_argument("--reducer", default="softmax", choices=["softmax", "mean", "max"])
    ap.add_argument("--softmax_tau", type=float, default=0.02)
    ap.add_argument("--kernel_px", type=float, default=1.2)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing chips.")
    ap.add_argument("--clip-lo", type=float, default=0.5, help="Global clip lo percentile (used if tiles scan fails).")
    ap.add_argument("--clip-hi", type=float, default=99.5, help="Global clip hi percentile (used if tiles scan fails).")
    ap.add_argument("--sample-stride", type=int, default=32, help="Stride when sampling global tiles for lo/hi.")
    args = ap.parse_args()

    root = args.root
    index_json = os.path.join(root, "index.json")
    if not os.path.isfile(index_json):
        raise FileNotFoundError(f"index.json not found at {index_json}. Run the global renderer first.")

    # Load global frame (mpp/tile_px/nx/ny/plane)
    gf = load_global_frame(index_json)

    # Compute absolute global lo/hi from your global tiles (fast sampling).
    tiles_glob = os.path.join(root, "tiles", "tile_*.npy")
    lo_global, hi_global = _approx_global_lohi_from_tiles(
        tiles_glob,
        clip_lo_pct=args.clip_lo,
        clip_hi_pct=args.clip_hi,
        sample_stride=max(1, args.sample_stride),
    )
    print(f"[scale] global lo/hi = {lo_global:.4f} / {hi_global:.4f}")

    # Pick submaps
    sm_root = os.path.join(root, "submaps")
    Path(sm_root).mkdir(parents=True, exist_ok=True)
    submap_dirs: List[str]
    if args.submap:
        submap_dirs = [os.path.join(sm_root, args.submap)]
    else:
        submap_dirs = [os.path.join(sm_root, d) for d in sorted(os.listdir(sm_root)) if os.path.isdir(os.path.join(sm_root, d))]

    if not submap_dirs:
        print(f"[chip] no submaps found under {sm_root}")
        return

    # Process each submap
    for sm_dir in submap_dirs:
        if not os.path.isdir(sm_dir):
            continue
        info = _chip_one_submap(
            root=root,
            sm_dir=sm_dir,
            gf=gf,
            lo_global=lo_global,
            hi_global=hi_global,
            reducer=args.reducer,
            softmax_tau=args.softmax_tau,
            kernel_px=args.kernel_px,
            overwrite=args.overwrite,
        )
        print(f"[chip] {info['submap_id']}: chips → {info['chips_dir']}  tiles={len(info['overlapped_tile_ids'])}")

    print("[chip] done.")

if __name__ == "__main__":
    main()
