#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend watcher:
- Re-chips submaps (unchanged).
- Optionally embeds every chip with windowed weighting:
    * For chip c in tile tid0, use ALL chips in that submap.
    * Weight each chip j by Gaussian of tile-grid distance from tid0.
    * Inside each chip, mask background (NaN/white) at patch level.
    * Resize each chip to --max-edge then pad to multiples of 14.

Produces one .embed.npy per chip (same count as chips in the submap).
"""

import os, time, argparse, traceback, shutil, glob, math, json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import cv2

# watchdog (optional)
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAVE_WATCHDOG = True
except Exception:
    HAVE_WATCHDOG = False

from vggt_slam.covis.submap_chipper import chip_submap_points

# -------------- DINO utils --------------
PATCH = 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

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
    H, W = valid_u8.shape
    H2 = (H // PATCH) * PATCH
    W2 = (W // PATCH) * PATCH
    m = valid_u8[:H2, :W2]
    m = m.reshape(H2//PATCH, PATCH, W2//PATCH, PATCH).mean(axis=(1,3))
    return (m > 0.0).astype(np.float32).reshape(-1)  # (P,)

@torch.no_grad()
def _extract_patch_tokens(model, u8: np.ndarray, device: str, use_amp: bool) -> torch.Tensor:
    u8p = _pad_to_multiple(u8, PATCH, 255)
    x = _prep_tensor(u8p, device)
    with torch.autocast('cuda', dtype=torch.float16, enabled=(use_amp and device.startswith("cuda"))):
        feat = model.forward_features(x)
    for k in ("x_norm_patchtokens", "x_norm_patch_tokens", "x_prenorm"):
        if isinstance(feat, dict) and (k in feat):
            return feat[k].squeeze(0)  # (P,D)
    if isinstance(feat, dict):
        for k in ("x_norm_clstoken", "x_norm_cls"):
            if k in feat:
                return feat[k].squeeze(0).unsqueeze(0)
    raise RuntimeError("Unexpected DINO features structure.")

def _read_index(root: str) -> Dict:
    idx = json.load(open(os.path.join(root, "index.json"), "r"))
    return {
        "nx": int(idx["nx"]),
        "ny": int(idx["ny"]),
        "tile_px": int(idx["tile_px"]),
        "viz_lo": float(idx.get("viz_lo", 0.0)),
        "viz_hi": float(idx.get("viz_hi", 1.0)),
    }

def _tid_to_xy(tid: int, nx: int) -> Tuple[int,int]:
    return (tid % nx, tid // nx)

def _tile_weight(dx: int, dy: int, sigma: float) -> float:
    return math.exp(-0.5 * (dx*dx + dy*dy) / (sigma*sigma))

# -------------- chipper I/O guards --------------
def _wait_for_ready(sm_dir: str, timeout: float = 30.0, poll: float = 0.1) -> bool:
    pts = os.path.join(sm_dir, "points_world.npy")
    ready = os.path.join(sm_dir, "READY")
    t0 = time.time()
    last_size, stable = -1, 0
    while time.time() - t0 < timeout:
        if os.path.isfile(pts):
            if os.path.isfile(ready):
                return True
            try:
                sz = os.path.getsize(pts)
                if sz == last_size and sz > 0:
                    stable += 1
                    if stable >= 5:
                        return True
                else:
                    stable = 0
                last_size = sz
            except FileNotFoundError:
                pass
        time.sleep(poll)
    return False

def _load_points_safe(npy_path: str, retries: int = 3, delay: float = 0.2):
    for _ in range(retries):
        try:
            arr = np.load(npy_path)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr.astype(np.float32, copy=False)
            if arr.ndim == 2 and arr.shape[0] == 3:
                return arr.T.astype(np.float32, copy=False)
            if arr.ndim == 1 and (arr.size % 3) == 0:
                return arr.reshape(-1, 3).astype(np.float32, copy=False)
        except Exception:
            time.sleep(delay)
    return None

# -------------- embeddings for submap chips --------------
@torch.no_grad()
def _embed_submap_chips_windowed(chips_dir: str, root: str,
                                 model_name: str,
                                 max_edge: int,
                                 sigma_tiles: float,
                                 device: str,
                                 use_amp: bool):
    """Embed every chip in the submap using all chips (weighted by global tile-grid distance)."""
    info = _read_index(root)
    nx = info["nx"]; viz_lo = info["viz_lo"]; viz_hi = info["viz_hi"]

    # Load model once
    model = load_dino(model_name).to(device)

    # list chips
    chip_paths = sorted(glob.glob(os.path.join(chips_dir, "*.npy")))
    if not chip_paths:
        print(f"[emb-submap] no chips found in {chips_dir}")
        return

    # map tid -> chip path
    tid2chip = {}
    for p in chip_paths:
        tid = int(os.path.splitext(os.path.basename(p))[0])
        tid2chip[tid] = p

    # cache token+mask per chip to avoid recompute
    cache: Dict[int, Dict] = {}

    # put near top (with other constants)
    MIN_VALID_PIXELS = 4  # hard floor on usable pixels per chip

    def get_entry(tid: int) -> Dict:
        """
        Load chip for tile id 'tid' and return tokens+mask entry.
        If a chip has too few valid pixels (< MIN_VALID_PIXELS), return {} so it contributes zero weight.
        """
        if tid in cache:
            return cache[tid]
        p = tid2chip.get(tid, None)
        if p is None:
            return {}

        # 1) load DEM and compute valid pixel count BEFORE any resize/pad
        dem = np.load(p).astype(np.float32)
        valid_pix = int(np.isfinite(dem).sum())
        if valid_pix < MIN_VALID_PIXELS:
            cache[tid] = {}  # mark unusable; treated as zero-weight / skipped
            return cache[tid]

        # 2) regular preprocessing (gray + mask), resize, pad, mask→patch grid
        u8, valid = _dem_to_u8_gray_with_white(dem, viz_lo, viz_hi)
        u8s = _resize_fit_max_edge(u8, max_edge)
        Hs, Ws = u8s.shape
        valid_s = cv2.resize(valid, (Ws, Hs), interpolation=cv2.INTER_NEAREST)

        u8p = _pad_to_multiple(u8s, PATCH, 255)
        Hp, Wp = u8p.shape
        valid_p = np.zeros((Hp, Wp), np.uint8)
        valid_p[:Hs, :Ws] = valid_s
        patch_mask = _mask_to_patch_grid(valid_p)  # (P,)

        tokens = _extract_patch_tokens(model, u8s, device, use_amp)  # (P,D)
        tokens = _normalize(tokens).cpu()

        cache[tid] = {"tokens": tokens, "mask": patch_mask}
        return cache[tid]


    # For each chip, compute an embedding using all chips in the submap
    for p in chip_paths:
        tid0 = int(os.path.splitext(os.path.basename(p))[0])
        out = p.replace(".npy", ".embed.npy")
        # Always overwrite in watcher
        tx0, ty0 = _tid_to_xy(tid0, nx)

        # accumulate weighted masked mean
        num = None
        den = 0.0
        for tid, pathc in tid2chip.items():
            entry = get_entry(tid)
            if not entry: 
                continue
            tx, ty = _tid_to_xy(tid, nx)
            w = _tile_weight(tx - tx0, ty - ty0, sigma_tiles)
            if w <= 1e-6:
                continue
            tok = entry["tokens"]
            msk = torch.from_numpy(entry["mask"]).to(tok.device).unsqueeze(1)  # (P,1)
            w_patch = w * msk
            if num is None:
                num = (tok * w_patch).sum(dim=0)
            else:
                num = num + (tok * w_patch).sum(dim=0)
            den += float(w_patch.sum().cpu().item())

        if den < 1e-8:
            # fallback to self-only masked mean
            e0 = get_entry(tid0)
            tok = e0["tokens"]; msk = torch.from_numpy(e0["mask"]).to(tok.device).unsqueeze(1)
            s = (tok * msk).sum(dim=0); z = float(msk.sum().cpu().item())
            v = s / max(z, 1e-8)
        else:
            v = num / den

        v = _normalize(v).cpu().numpy().astype(np.float32)
        np.save(out, v)

    print(f"[emb-submap] wrote {len(chip_paths)} embeddings in {chips_dir}")

# -------------- watcher workflow --------------
def process_one_submap(root: str, sm_dir: str,
                       do_embeddings: bool,
                       dino_model: str,
                       max_edge: int,
                       sigma_tiles: float):
    index_json = os.path.join(root, "index.json")
    if not os.path.isfile(index_json):
        print(f"[watch] {os.path.basename(sm_dir)}: index.json missing — retry later.")
        return
    if not _wait_for_ready(sm_dir):
        print(f"[watch] {os.path.basename(sm_dir)}: waiting for READY/points; skip.")
        return

    pts_file = os.path.join(sm_dir, "points_world.npy")
    P_world = _load_points_safe(pts_file)
    if P_world is None or P_world.size == 0:
        print(f"[watch] {os.path.basename(sm_dir)}: no usable points; retry later.")
        return

    chips_dir = os.path.join(sm_dir, "chips")
    if os.path.isdir(chips_dir):
        shutil.rmtree(chips_dir, ignore_errors=True)
    Path(chips_dir).mkdir(parents=True, exist_ok=True)

    # chip
    try:
        info = chip_submap_points(
            submap_id=os.path.basename(sm_dir),
            P_world=P_world,
            out_dir=root,
            index_json=index_json,
            reducer="softmax",
            softmax_tau=0.02,
            kernel_px=1.2,
        )
        n_chips = len(glob.glob(os.path.join(chips_dir, "*.npy")))
        print(f"[watch] chipped {os.path.basename(sm_dir)}: {n_chips} chips, {len(info.get('overlapped_tile_ids', []))} tiles")
    except Exception:
        print(f"[watch] error while chipping {sm_dir}")
        traceback.print_exc()
        return

    # windowed embeddings (optional)
    if do_embeddings:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = True
        try:
            print(f"[watch] embedding (windowed) for {os.path.basename(sm_dir)} using {dino_model} …")
            _embed_submap_chips_windowed(
                chips_dir=chips_dir, root=root,
                model_name=dino_model, max_edge=max_edge,
                sigma_tiles=sigma_tiles, device=device, use_amp=use_amp
            )
        except Exception:
            print(f"[watch] embedding error for {os.path.basename(sm_dir)}")
            traceback.print_exc()

def batch_process_existing(root: str, do_embeddings: bool, dino_model: str, max_edge: int, sigma_tiles: float):
    sm_root = os.path.join(root, "submaps")
    if not os.path.isdir(sm_root):
        print(f"[watch] no submaps directory yet at {sm_root}; nothing to do.")
        return
    for d in sorted(os.listdir(sm_root)):
        sm_dir = os.path.join(sm_root, d)
        if not os.path.isdir(sm_dir):
            continue
        try:
            process_one_submap(root, sm_dir, do_embeddings, dino_model, max_edge, sigma_tiles)
        except Exception:
            print(f"[watch] error while processing {sm_dir}")
            traceback.print_exc()

class SubmapHandler(FileSystemEventHandler):
    def __init__(self, root, do_embeddings, dino_model, max_edge, sigma_tiles):
        self.root = root
        self.do_embeddings = do_embeddings
        self.dino_model = dino_model
        self.max_edge = max_edge
        self.sigma_tiles = sigma_tiles

    def _try_process(self, path: str):
        try:
            sm_dir = os.path.dirname(path)
            if os.path.basename(os.path.dirname(sm_dir)) != "submaps":
                return
            pts = os.path.join(sm_dir, "points_world.npy")
            if os.path.isfile(pts):
                process_one_submap(self.root, sm_dir, self.do_embeddings, self.dino_model, self.max_edge, self.sigma_tiles)
        except Exception:
            traceback.print_exc()

    def on_created(self, event):
        if event.is_directory:
            if os.path.basename(os.path.dirname(event.src_path)) == "submaps":
                print(f"[watch] new submap dir: {event.src_path}")
            return
        if event.src_path.endswith("points_world.npy") or event.src_path.endswith("READY"):
            self._try_process(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith("points_world.npy") or event.src_path.endswith("READY"):
            self._try_process(event.src_path)

def main():
    ap = argparse.ArgumentParser("Backend watcher: ALWAYS re-chip submaps (+ optional windowed DINO embeddings)")
    ap.add_argument("--root", required=True, help="Path to global DEM root (contains index.json and submaps/)")
    ap.add_argument("--with-embeddings", action="store_true")
    ap.add_argument("--dino-model", default="facebook/dinov2-base")
    ap.add_argument("--rescan-every", type=float, default=300.0, help="full rescan every N seconds")
    ap.add_argument("--max-edge", type=int, default=1024, help="resize bound for chip embeddings (speed)")
    ap.add_argument("--sigma-tiles", type=float, default=2.0, help="Gaussian sigma in tile units")
    args = ap.parse_args()

    batch_process_existing(args.root, args.with_embeddings, args.dino_model, args.max_edge, args.sigma_tiles)

    sm_root = os.path.join(args.root, "submaps")
    Path(sm_root).mkdir(parents=True, exist_ok=True)

    if HAVE_WATCHDOG:
        obs = Observer()
        handler = SubmapHandler(args.root, args.with_embeddings, args.dino_model, args.max_edge, args.sigma_tiles)
        obs.schedule(handler, sm_root, recursive=True)
        obs.start()
        print(f"[watch] using watchdog; watching {sm_root} … (Ctrl+C to stop)")
        try:
            t0 = time.time()
            while True:
                time.sleep(1.0)
                if time.time() - t0 >= args.rescan_every:
                    batch_process_existing(args.root, args.with_embeddings, args.dino_model, args.max_edge, args.sigma_tiles)
                    t0 = time.time()
        except KeyboardInterrupt:
            obs.stop()
        obs.join()
    else:
        print(f"[watch] polling (no watchdog) …")
        try:
            while True:
                batch_process_existing(args.root, args.with_embeddings, args.dino_model, args.max_edge, args.sigma_tiles)
                time.sleep(args.rescan_every)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
