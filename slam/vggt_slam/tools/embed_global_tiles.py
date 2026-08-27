#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed global tiles with 9×9 windowed weighting + background masking.
- One embedding per tile (processes ALL tiles).
- Neighborhood radius configurable (default 4 => 9x9).
- Background (NaN/white) is masked out at the patch level.
- Image is resized to --max-edge then padded to multiples of patch=14 (ViT-B/14).
- Writes embeddings to global_embeddings/ and a valid.json listing embedded (valid) tile IDs.

CLI (compatible):
  python vggt_slam/tools/embed_global_tiles.py \
    --root outputs/05 \
    --dino-model facebook/dinov2-base \
    --mode resize --max-edge 1536
"""

import os, json, glob, argparse, math
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import cv2
import re

torch.backends.cudnn.benchmark = True

PATCH = 14  # ViT-B/14

# ---------------- DINO loader ----------------
def _normalize(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, dim=-1)

def load_dino(model_name: str):
    alias = {
        "facebook/dinov2-base": "dinov2_vitb14",
        "dinov2-base": "dinov2_vitb14",
        "vitb14": "dinov2_vitb14",
    }
    name = alias.get(model_name.lower(), model_name)
    print(f"[dino] loading: {model_name} → {name}")
    mdl = torch.hub.load('facebookresearch/dinov2', name).eval()
    return mdl

# ---------------- helpers ----------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def _pad_to_multiple(u8: np.ndarray, k: int, pad_value: int = 255) -> np.ndarray:
    H, W = u8.shape[:2]
    H2 = ( (H + k - 1) // k ) * k
    W2 = ( (W + k - 1) // k ) * k
    if H2 == H and W2 == W:
        return u8
    out = np.full((H2, W2), pad_value, dtype=np.uint8)
    out[:H, :W] = u8
    return out

def _to_chw3_u8(u8: np.ndarray) -> np.ndarray:
    return np.stack([u8, u8, u8], axis=0)

def _prep_tensor(u8: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(_to_chw3_u8(u8)).float().div_(255.0).unsqueeze(0)  # 1x3xHxW
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device, non_blocking=True)

def _dem_to_u8_gray_with_white(dem: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (u8_gray, valid_mask_u8). NaN→white; valid mask True where finite."""
    mask = np.isfinite(dem)
    u8 = np.full(dem.shape, 255, np.uint8)
    if mask.any():
        g = (np.clip(dem[mask], lo, hi) - lo) / (hi - lo + 1e-12)
        u8[mask] = (g * 255.0 + 0.5).astype(np.uint8)
    return u8, mask.astype(np.uint8)

def _resize_fit_max_edge(u8: np.ndarray, max_edge: int) -> np.ndarray:
    H, W = u8.shape
    if max(H, W) <= max_edge:
        return u8
    scale = float(max_edge) / float(max(H, W))
    Ht = max(1, int(round(H * scale)))
    Wt = max(1, int(round(W * scale)))
    return cv2.resize(u8, (Wt, Ht), interpolation=cv2.INTER_AREA)

def _mask_to_patch_grid(valid_u8: np.ndarray) -> np.ndarray:
    """Downsample a HxW binary mask to patch-grid (H/14 x W/14) via average>0."""
    H, W = valid_u8.shape
    H2 = (H // PATCH) * PATCH
    W2 = (W // PATCH) * PATCH
    if H2 == 0 or W2 == 0:
        return np.zeros((1,1), np.float32)
    m = valid_u8[:H2, :W2]
    m = m.reshape(H2//PATCH, PATCH, W2//PATCH, PATCH).mean(axis=(1,3))
    return (m > 0.0).astype(np.float32)  # 1 where any valid pixels in the patch

def _read_index(root: str) -> Dict:
    idx = json.load(open(os.path.join(root, "index.json"), "r"))
    # viz_lo/hi are required to match grayscale
    viz_lo = float(idx.get("viz_lo", 0.0))
    viz_hi = float(idx.get("viz_hi", 1.0))
    return {
        "nx": int(idx["nx"]),
        "ny": int(idx["ny"]),
        "tile_px": int(idx["tile_px"]),
        "viz_lo": viz_lo,
        "viz_hi": viz_hi,
    }

def _tid_to_xy(tid: int, nx: int) -> Tuple[int,int]:
    return (tid % nx, tid // nx)

# Gaussian weight based on tile-grid distance
def _tile_weight(dx: int, dy: int, sigma: float) -> float:
    return math.exp(-0.5 * (dx*dx + dy*dy) / (sigma*sigma))

# Extract patch tokens at controlled input size (<=max_edge), pad to multiples of 14.
@torch.no_grad()
def _extract_patch_tokens(model, u8: np.ndarray, device: str, use_amp: bool) -> torch.Tensor:
    u8p = _pad_to_multiple(u8, PATCH, 255)
    x = _prep_tensor(u8p, device)
    with torch.autocast('cuda', dtype=torch.float16, enabled=(use_amp and device.startswith("cuda"))):
        feat = model.forward_features(x)
    # Prefer patch tokens
    for k in ("x_norm_patchtokens", "x_norm_patch_tokens", "x_prenorm"):
        if isinstance(feat, dict) and (k in feat):
            return feat[k].squeeze(0)  # (P, D)
    # fallback to CLS
    if isinstance(feat, dict):
        for k in ("x_norm_clstoken", "x_norm_cls"):
            if k in feat:
                return feat[k].squeeze(0).unsqueeze(0)  # (1, D)
    raise RuntimeError("Unexpected DINO features structure.")

# ---------------- windowed embedding (global) ----------------
def main():
    ap = argparse.ArgumentParser("Embed global tiles with 9x9 windowed weighting + background mask")
    ap.add_argument("--root", required=True, help="path containing tiles/ and index.json")
    ap.add_argument("--dino-model", default="facebook/dinov2-base")
    ap.add_argument("--overwrite", action="store_true")

    # Window parameters (default → 9x9)
    ap.add_argument("--window-radius", type=int, default=4, help="tiles each side (4→9x9)")
    ap.add_argument("--sigma-tiles", type=float, default=2.0, help="Gaussian sigma in tile units")

    # Skip tiles that are basically empty
    ap.add_argument("--min-valid-frac", type=float, default=0.01,
                    help="skip tiles with < this fraction of finite pixels")

    # Speed/compatibility knobs (kept; respected)
    ap.add_argument("--mode", default="resize")           # kept for compat
    ap.add_argument("--max-edge", type=int, default=1536) # respected to bound input
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--min-valid-pixels", type=int, default=20,
                help="hard floor on usable pixels per tile; tiles with < this many valid pixels are ignored")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (not args.no_amp)

    info = _read_index(args.root)
    nx, ny, tile_px = info["nx"], info["ny"], info["tile_px"]
    viz_lo, viz_hi = info["viz_lo"], info["viz_hi"]


    # list tiles and make (tid -> path) map
    tile_glob = os.path.join(args.root, "tiles", "tile_*.npy")
    paths = sorted(glob.glob(tile_glob))
    # 1) ignore embedding files (tile_XXXXX.embed.npy)
    paths = [p for p in paths if not p.endswith(".embed.npy")]

    tid2path: Dict[int, str] = {}
    rx = re.compile(r"^tile_(\d+)\.npy$")
    for p in paths:
        fname = os.path.basename(p)
        m = rx.match(fname)
        if not m:
            # skip anything that isn't exactly tile_<digits>.npy
            continue
        tid = int(m.group(1))
        tid2path[tid] = p


    if not tid2path:
        print("[emb] no tiles found.")
        return

    model = load_dino(args.dino_model).to(device)

    # cache per-tile { "tokens": torch.Tensor(P,D), "mask": np.ndarray(P,), "Hpatch": int, "Wpatch": int }
    cache: Dict[int, Dict] = {}

    def get_cached_tokens(tid: int) -> Dict:
        if tid in cache:
            return cache[tid]
        p = tid2path.get(tid, None)
        if p is None:
            return {}

        dem = np.load(p).astype(np.float32)
        finite = np.isfinite(dem)

        # Hard absolute count gate (new) + existing fractional gate
        if int(finite.sum()) < args.min_valid_pixels or finite.mean() < args.min_valid_frac:
            cache[tid] = {}   # mark unusable: contributes zero weight
            return cache[tid]

        u8, valid = _dem_to_u8_gray_with_white(dem, viz_lo, viz_hi)
        # downscale to max-edge then pad to multiple of 14
        u8s = _resize_fit_max_edge(u8, args.max_edge)
        Hs, Ws = u8s.shape
        valid_s = cv2.resize(valid, (Ws, Hs), interpolation=cv2.INTER_NEAREST)

        # build patch mask on the image grid first (multiple of 14)
        u8p = _pad_to_multiple(u8s, PATCH, 255)
        Hp, Wp = u8p.shape
        valid_p = np.zeros((Hp, Wp), np.uint8)
        valid_p[:Hs, :Ws] = valid_s
        patch_mask = _mask_to_patch_grid(valid_p).reshape(-1)  # (P,)

        # extract patch tokens
        tokens = _extract_patch_tokens(model, u8s, device, use_amp)  # (P,D)
        tokens = _normalize(tokens).cpu()

        cache[tid] = {
            "tokens": tokens,   # torch (P,D)
            "mask": patch_mask, # np (P,)
        }
        return cache[tid]

    out_dir = os.path.join(args.root, "global_embeddings")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    valid_ids = []
    total = len(tid2path)
    done = 0

    for tid, p in sorted(tid2path.items()):
        out = os.path.join(out_dir, f"tile_{tid:05d}.embed.npy")
        if (not args.overwrite) and os.path.isfile(out):
            valid_ids.append(tid)
            done += 1
            continue

        tx, ty = _tid_to_xy(tid, nx)
        # collect window neighbors
        tiles_win = []
        for dy in range(-args.window_radius, args.window_radius+1):
            for dx in range(-args.window_radius, args.window_radius+1):
                nx_ = tx + dx; ny_ = ty + dy
                if nx_ < 0 or nx_ >= nx or ny_ < 0 or ny_ >= ny:
                    continue  # edge-aware: shrink window near borders
                tid_n = ny_ * nx + nx_
                entry = get_cached_tokens(tid_n)
                if not entry:  # empty / invalid neighbor
                    continue
                w_tile = _tile_weight(dx, dy, args.sigma_tiles)
                if w_tile <= 1e-6:
                    continue
                tiles_win.append((entry, w_tile))

        if not tiles_win:
            # center tile itself may be invalid; skip entirely
            done += 1
            continue

        # weighted masked mean over patch tokens across all neighbors
        num = None
        den = 0.0
        for entry, w in tiles_win:
            tok = entry["tokens"]            # (P,D) torch
            msk = torch.from_numpy(entry["mask"]).to(tok.device).unsqueeze(1)  # (P,1)
            w_patch = w * msk                # (P,1) float weights (0 where invalid)
            if num is None:
                num = (tok * w_patch).sum(dim=0)
            else:
                num = num + (tok * w_patch).sum(dim=0)
            den += float(w_patch.sum().cpu().item())

        if den < 1e-8:
            # all patches ended up invalid; skip
            done += 1
            continue

        v = num / den
        v = _normalize(v).cpu().numpy().astype(np.float32)
        np.save(out, v)
        valid_ids.append(tid)
        done += 1
        if done % 25 == 0 or done == total:
            print(f"[emb] {done}/{total}")

    # write manifest of valid (embedded) tiles
    with open(os.path.join(out_dir, "valid.json"), "w") as f:
        json.dump({"valid_ids": sorted(valid_ids)}, f, indent=2)

    # write a simple index.json listing vectors (optional, for tooling)
    idx = [{"tile": f"tile_{tid:05d}.npy",
            "embed": f"tile_{tid:05d}.embed.npy"} for tid in sorted(valid_ids)]
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"vectors": idx,
                   "model": args.dino_model,
                   "window_radius": args.window_radius,
                   "sigma_tiles": args.sigma_tiles,
                   "max_edge": args.max_edge,
                   "min_valid_frac": args.min_valid_frac}, f, indent=2)

    print(f"[emb] done: wrote {len(valid_ids)} embeddings → {out_dir}")
    print(f"[emb] valid.json created with {len(valid_ids)} tile IDs")
if __name__ == "__main__":
    main()
