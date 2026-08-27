from __future__ import annotations
import os, glob, json
import numpy as np
from pathlib import Path
from .embed_from_dem import DEMEmbedder

def _dem_to_u8_gray(dem: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if dem.size == 0:
        return np.zeros((1,1), np.uint8)
    vmin, vmax = np.percentile(dem, [lo, hi])
    if abs(vmax - vmin) < 1e-12:
        return np.zeros_like(dem, np.uint8)
    t = (dem - vmin) / (vmax - vmin)
    t = np.clip(t, 0, 1)
    return (t * 255.0 + 0.5).astype(np.uint8)

def _read_clips(root_index_json: str) -> tuple[float,float]:
    with open(root_index_json, "r") as f:
        j = json.load(f)
    return float(j.get("clip_lo", 1.0)), float(j.get("clip_hi", 99.0))

def embed_tiles(root_dir: str, model_name: str = "facebook/dinov2-base"):
    """
    root_dir: e.g. outputs/run
    Will read tiles/*.npy and write tiles/*.embed.npy using SAME clipping as index.json.
    """
    emb = DEMEmbedder(model_name=model_name)
    index_json = os.path.join(root_dir, "index.json")
    clip_lo, clip_hi = _read_clips(index_json)

    npys = sorted(glob.glob(os.path.join(root_dir, "tiles", "tile_*.npy")))
    for p in npys:
        out = p.replace(".npy", ".embed.npy")
        if os.path.exists(out):
            continue
        dem = np.load(p).astype(np.float32)
        u8 = _dem_to_u8_gray(dem, clip_lo, clip_hi)
        vec = emb.embed_uint8_patch(u8)
        np.save(out, vec.astype(np.float32))

def embed_submap_chips(chips_dir: str, model_name: str = "facebook/dinov2-base"):
    """
    chips_dir: outputs/run/submaps/sm_xxxxx/chips
    Uses the SAME clipping numbers stored in submap meta.json (which mirrors index.json).
    """
    # infer submap meta and read clip settings
    sm_dir = str(Path(chips_dir).parent)
    with open(os.path.join(sm_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    clip_lo = float(meta.get("clip_lo", 1.0))
    clip_hi = float(meta.get("clip_hi", 99.0))

    emb = DEMEmbedder(model_name=model_name)
    npys = sorted(glob.glob(os.path.join(chips_dir, "*.npy")))
    for p in npys:
        if p.endswith(".embed.npy"):
            continue
        out = p.replace(".npy", ".embed.npy")
        if os.path.exists(out):
            continue
        dem = np.load(p).astype(np.float32)
        u8 = _dem_to_u8_gray(dem, clip_lo, clip_hi)
        vec = emb.embed_uint8_patch(u8)
        np.save(out, vec.astype(np.float32))
