from __future__ import annotations
import os, json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import cv2

from .frame_io import load_global_frame, world_to_plane, plane_to_tile_idx, tile_bbox
# import the exact same rasterizer + visualizer the global renderer uses
from vggt_slam.global_dem_tiled import _rasterize_tile, _grayscale_viz_from_dem


def _read_viz_and_params(index_json: str) -> Tuple[Tuple[float, float], dict]:
    """Read absolute lo/hi and visual params written by the global renderer."""
    d = json.load(open(index_json, "r"))
    # absolute grayscale scale used globally
    viz_lo = float(d.get("viz_lo", 0.0))
    viz_hi = float(d.get("viz_hi", 1.0))
    # visual pipeline knobs (edge/shade/dark/unsharp/clahe)
    vp = d.get("visual_params", {})
    params = {
        "edge_strength": float(vp.get("edge_strength", 0.95)),
        "shade_strength": float(vp.get("shade_strength", 0.70)),
        "dark_level": float(vp.get("dark_level", 0.09)),
        "unsharp_radius_px": float(vp.get("unsharp_radius_px", 1.0)),
        "unsharp_amount": float(vp.get("unsharp_amount", 0.8)),
        "clahe_clip": float(vp.get("clahe_clip", 3.0)),
        "clahe_grid": int(vp.get("clahe_grid", 8)),
        # we keep the same gamma the global path uses
        "gamma": 0.85,
    }
    return (viz_lo, viz_hi), params


def chip_submap_points(
    submap_id: str,
    P_world: np.ndarray,   # (N,3)
    out_dir: str,          # e.g. outputs/run
    index_json: str,       # e.g. outputs/run/index.json
    reducer: str = "softmax",
    softmax_tau: float = 0.02,
    kernel_px: float = 1.2,
) -> Dict:
    """
    Writes:
      <out_dir>/submaps/<submap_id>/chips/<tile_id>.npy
      <out_dir>/submaps/<submap_id>/chips/<tile_id>.png  (white bg preview, identical look to global)
      <out_dir>/submaps/<submap_id>/tiles.json
      <out_dir>/submaps/<submap_id>/meta.json
    """
    gf = load_global_frame(index_json)   # provides N,R,o,mpp,tile_px,nx,ny,u0,v0
    sub_dir = os.path.join(out_dir, "submaps", submap_id)
    chips_dir = os.path.join(sub_dir, "chips")
    Path(chips_dir).mkdir(parents=True, exist_ok=True)

    if P_world.size == 0:
        raise RuntimeError(f"{submap_id}: no points to chip.")

    # Same projection and height as global renderer
    uv = world_to_plane(P_world, gf)   # (N,2)
    Z = (P_world @ gf.N.astype(np.float64)) + float(gf.d)
    Z = Z.astype(np.float32)

    # Overlapping tile range
    sx0, sy0 = uv.min(axis=0).tolist()
    sx1, sy1 = uv.max(axis=0).tolist()
    tx0, ty0 = plane_to_tile_idx(sx0, sy0, gf)
    tx1, ty1 = plane_to_tile_idx(sx1, sy1, gf)
    tx0 = max(0, min(gf.nx - 1, tx0)); tx1 = max(0, min(gf.nx - 1, tx1))
    ty0 = max(0, min(gf.ny - 1, ty0)); ty1 = max(0, min(gf.ny - 1, ty1))

    # Persist metadata (useful for debugging)
    with open(os.path.join(sub_dir, "meta.json"), "w") as fmeta:
        json.dump({
            "sm_id": submap_id,
            "plane_bbox": [float(sx0), float(sy0), float(sx1), float(sy1)],
            "mpp": float(gf.mpp),
            "tile_px": int(gf.tile_px),
            "frame_index_json": os.path.abspath(index_json),
            "reducer": reducer,
            "softmax_tau": float(softmax_tau),
            "kernel_px": float(kernel_px),
        }, fmeta, indent=2)

    # Use the SAME absolute grayscale scale + visual knobs as global tiles
    (viz_lo, viz_hi), vp = _read_viz_and_params(index_json)

    overlapped_tiles: List[int] = []
    for ty in range(min(ty0,ty1), max(ty0,ty1)+1):
        for tx in range(min(tx0,tx1), max(tx0,tx1)+1):
            tb = tile_bbox(tx, ty, gf)  # (u0,v0,u1,v1) meters
            u0,v0,u1,v1 = tb
            inside = (uv[:,0] >= u0) & (uv[:,0] < u1) & (uv[:,1] >= v0) & (uv[:,1] < v1)
            if not np.any(inside):
                continue

            # Force identical raster size as the global tiles
            dem_chip, occ_chip = _rasterize_tile(
                uv[inside], Z[inside], tb,
                gf.mpp, kernel_px, reducer, softmax_tau,
                tile_px_fixed=gf.tile_px,  # requires the public arg exposed in your renderer
            )

            tid = ty * gf.nx + tx
            overlapped_tiles.append(int(tid))

            # Save DEM chip
            np.save(os.path.join(chips_dir, f"{tid:05d}.npy"), dem_chip.astype(np.float32))

            # Create PNG preview using the SAME visualization pipeline
            img_bgr = _grayscale_viz_from_dem(
                dem=dem_chip,
                mpp=gf.mpp,
                edge_strength=vp["edge_strength"],
                shade_strength=vp["shade_strength"],
                dark_level=vp["dark_level"],
                unsharp_radius_px=vp["unsharp_radius_px"],
                unsharp_amount=vp["unsharp_amount"],
                clahe_clip=vp["clahe_clip"],
                clahe_grid=vp["clahe_grid"],
                lo_hi=(viz_lo, viz_hi),
                gamma=vp["gamma"],
            )
            # guarantee white outside valid DEM (the helper already does this)
            cv2.imwrite(os.path.join(chips_dir, f"{tid:05d}.png"), img_bgr)

    with open(os.path.join(sub_dir, "tiles.json"), "w") as f:
        json.dump({"overlapped_tile_ids": overlapped_tiles}, f, indent=2)

    return {
        "submap_id": submap_id,
        "overlapped_tile_ids": overlapped_tiles,
        "chips_dir": chips_dir
    }
