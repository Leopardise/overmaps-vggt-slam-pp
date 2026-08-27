from __future__ import annotations

import os, re, glob, json, argparse, traceback, contextlib
from typing import Tuple, Optional, List
import numpy as np
import cv2
import torch

# --------------------- helpers ---------------------

def _read_viz_scale(index_json: str) -> Tuple[float, float]:
    d = json.load(open(index_json, "r"))
    lo = float(d.get("viz_lo", 0.0))
    hi = float(d.get("viz_hi", 1.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    return lo, hi


def dem_to_u8_gray_global(dem: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map DEM (float32, NaNs allowed) → uint8 gray; NaNs become white (255)."""
    u8 = np.empty(dem.shape, np.uint8)
    mask = np.isfinite(dem)
    if mask.any():
        x = np.clip(dem[mask], lo, hi)
        g = (x - lo) / (hi - lo + 1e-12)
        u8[mask] = (g * 255.0 + 0.5).astype(np.uint8)
    u8[~mask] = 255
    return u8


def map_model_name(name: str) -> str:
    name = (name or "").strip().lower()
    if name in ["facebook/dinov2-base", "dinov2-base", "dinov2_vitb14", "vitb14"]:
        return "dinov2_vitb14"
    if name in ["facebook/dinov2-large", "dinov2-large", "dinov2_vitl14", "vitl14"]:
        return "dinov2_vitl14"
    if name in ["facebook/dinov2-giant", "dinov2-giant", "dinov2_vitg14", "vitg14"]:
        return "dinov2_vitg14"
    return "dinov2_vitb14"


@torch.no_grad()
def dino_embed_uint8_basic(
    model,
    u8: np.ndarray,
    device: str,
    mode: str = "resize",
    patch: int = 14,
    max_edge: int = 1536,
    use_amp: bool = True,
) -> np.ndarray:
    """Compute a normalized DINO CLS embedding from a uint8 grayscale chip."""
    if u8.ndim != 2:
        raise ValueError("chip must be HxW uint8 (grayscale)")

    H, W = u8.shape
    rgb = np.repeat(u8[..., None], 3, axis=2)

    # Prefer resize to match global tile preprocessing
    if mode == "resize" and max(H, W) > max_edge:
        if H >= W:
            newH = max_edge
            newW = int(round(W * (max_edge / H)))
        else:
            newW = max_edge
            newH = int(round(H * (max_edge / W)))
        rgb = cv2.resize(rgb, (newW, newH), interpolation=cv2.INTER_AREA)
        H, W = newH, newW

    # Pad-center to a multiple of patch size
    Hc = max(patch, (H // patch) * patch)
    Wc = max(patch, (W // patch) * patch)
    ph = max(0, Hc - H)
    pw = max(0, Wc - W)
    if ph or pw:
        rgb = cv2.copyMakeBorder(
            rgb, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_REFLECT_101
        )
        H, W = rgb.shape[:2]

    y0 = (H - Hc) // 2
    x0 = (W - Wc) // 2
    rgb = rgb[y0 : y0 + Hc, x0 : x0 + Wc]

    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = x.to(device, non_blocking=True)

    use_cuda = device.startswith("cuda")
    if use_cuda:
        ac = torch.cuda.amp.autocast
        dtype = torch.float16
    else:
        # Older torch builds may lack torch.cpu.amp.autocast → nullcontext
        ac = getattr(torch.cpu.amp, "autocast", lambda **_: contextlib.nullcontext())
        dtype = torch.float32

    with ac(enabled=use_amp):
        feat = model.forward_features(x)
        cls = feat.get("x_norm_clstoken") or feat.get("x_norm_cls")
        if cls is None:
            raise RuntimeError("DINO features missing CLS token")
        vec = cls.detach().flatten(1).to(dtype=dtype)

    v = vec.squeeze(0).float().cpu().numpy()
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float32)


def load_dino(model_name: str, device: str):
    m = map_model_name(model_name)
    print(f"[dino] loading: {model_name} → {m}")
    mdl = torch.hub.load("facebookresearch/dinov2", m).eval().to(device)
    return mdl, (14 if "14" in m else 16)


def load_faiss(root: str, which: str = "hnsw"):
    import faiss

    idx_p = os.path.join(root, "global_embeddings", f"faiss_{which}.index")
    idmap_p = os.path.join(root, "global_embeddings", "idmap.json")
    if not os.path.isfile(idx_p) or not os.path.isfile(idmap_p):
        raise FileNotFoundError(
            "FAISS index or idmap missing; run vggt_slam/tools/faiss_global_index.py first."
        )
    index = faiss.read_index(idx_p)
    ids = json.load(open(idmap_p, "r"))["ids"]
    return index, ids


def _load_valid_ids(root: str) -> Optional[set]:
    """Tiles considered 'valid' (e.g., non-background) if file exists."""
    valid_p = os.path.join(root, "global_embeddings", "valid.json")
    if os.path.isfile(valid_p):
        try:
            return set(json.load(open(valid_p, "r")).get("valid_ids", []))
        except Exception:
            pass
    return None


def _chip_id_from_path(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r"(\d+)\.npy$", base)
    if m:
        return int(m.group(1))
    m = re.search(r"chip[_-]?(\d+)\.npy$", base, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser("Submap chip→FAISS match vs global tiles (robust)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--submap", required=True)  # e.g., sm_00012 or 'latest'
    ap.add_argument("--faiss-index", default="hnsw", choices=["hnsw", "flatip"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--dino-model", default="facebook/dinov2-base")
    ap.add_argument("--mode", default="resize", choices=["crop", "resize"])
    ap.add_argument("--max-edge", type=int, default=1536)
    ap.add_argument("--save-chips", action="store_true")
    ap.add_argument("--exclude-self", action="store_true")

    # Validity thresholds for DEM chips (either condition triggers skip)
    ap.add_argument(
        "--min-chip-valid-frac",
        type=float,
        default=0.01,
        help="skip chip if finite.mean() < this",
    )
    ap.add_argument(
        "--min-chip-valid-count",
        type=int,
        default=4,
        help="skip chip if #finite < this",
    )

    # Overfetch before filtering to avoid empty keeps
    ap.add_argument(
        "--overfetch",
        type=int,
        default=5,
        help="fetch topk*overfetch from FAISS before post-filtering",
    )

    # Optional heatmap visualization
    ap.add_argument("--vis", action="store_true")

    args = ap.parse_args()

    # Locate submap & chips
    sm_root = os.path.join(args.root, "submaps")
    subs = sorted([d for d in os.listdir(sm_root) if d.startswith("sm_")])
    if not subs:
        raise SystemExit("no submaps found in <root>/submaps")

    sm_id = subs[-1] if args.submap == "latest" else args.submap
    sm_dir = os.path.join(sm_root, sm_id)
    chips_dir = os.path.join(sm_dir, "chips")
    if not os.path.isdir(chips_dir):
        raise SystemExit(f"chips dir missing: {chips_dir} (run backend_watch.py)")

    # Owners for optional self-exclusion
    owners = {}
    owners_p = os.path.join(args.root, "submap_index", "tile_owners.json")
    if args.exclude_self:
        if not os.path.isfile(owners_p):
            raise SystemExit(
                f"--exclude-self requested but {owners_p} not found. Run build_tile_owners.py"
            )
        owners = json.load(open(owners_p, "r")).get("tile_owners", {})

    # Global viz scale + grid dims
    index_json = os.path.join(args.root, "index.json")
    meta = json.load(open(index_json, "r"))
    viz_lo, viz_hi = _read_viz_scale(index_json)
    nx, ny = int(meta["nx"]), int(meta["ny"])

    # Optional valid-tiles mask
    valid_ids = _load_valid_ids(args.root)

    # FAISS & DINO
    index, idmap = load_faiss(args.root, which=args.faiss_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, patch = load_dino(args.dino_model, device)

    chip_npys = sorted(
        p for p in glob.glob(os.path.join(chips_dir, "*.npy"))
        if not p.endswith(".embed.npy")
    )
    print(f"[chips] {sm_id}: {len(chip_npys)} DEM chips")

    matches = []
    import faiss  # noqa: F401 (kept to surface import errors early)

    # Debug counters
    n_skipped_too_sparse = 0
    n_empty_after_valid = 0
    n_empty_after_self = 0

    for i, p in enumerate(chip_npys, 1):
        dem = np.load(p).astype(np.float32)
        finite = np.isfinite(dem)
        valid_frac = float(finite.mean())
        valid_count = int(finite.sum())
        if (valid_frac < args.min_chip_valid_frac) or (valid_count < args.min_chip_valid_count):
            n_skipped_too_sparse += 1
            continue

        # Prefer cached chip embedding; else compute
        pre = p.replace(".npy", ".embed.npy")
        if os.path.isfile(pre):
            vec = np.load(pre).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        else:
            u8 = dem_to_u8_gray_global(dem, viz_lo, viz_hi)
            if args.save_chips:
                cv2.imwrite(p.replace(".npy", ".png"), np.repeat(u8[..., None], 3, axis=2))
            vec = dino_embed_uint8_basic(
                model, u8, device, mode=args.mode, patch=patch, max_edge=args.max_edge
            )

        # FAISS search (overfetch to survive filtering)
        overK = max(args.topk * args.overfetch, args.topk)
        q = vec.reshape(1, -1).astype(np.float32)
        D, I = index.search(q, overK)
        tids = [int(idmap[j]) for j in I[0]]
        scores = [float(s) for s in D[0]]

        # Filtering stages
        def filter_hits(allow_invalid_ids: bool, allow_self: bool) -> List[tuple[int, float]]:
            keep: List[tuple[int, float]] = []
            for tid, sc in zip(tids, scores):
                if (valid_ids is not None) and (not allow_invalid_ids) and (tid not in valid_ids):
                    continue
                if args.exclude_self and (not allow_self):
                    claimers = owners.get(str(tid), [])
                    if sm_id in claimers:
                        continue
                keep.append((tid, sc))
                if len(keep) == args.topk:
                    break
            return keep

        keep = filter_hits(allow_invalid_ids=False, allow_self=False)

        if not keep:
            keep = filter_hits(allow_invalid_ids=True, allow_self=False)
            if not keep:
                n_empty_after_valid += 1

        if not keep:
            keep = filter_hits(allow_invalid_ids=True, allow_self=True)
            if not keep:
                n_empty_after_self += 1

        if keep:
            chip_id = _chip_id_from_path(p) or i  # fallback to ordinal
            matches.append(
                {
                    "chip_id": int(chip_id),
                    "top_ids": [tid for tid, _ in keep],
                    "scores": [sc for _, sc in keep],
                }
            )

        if (i % 10 == 0) or (i == len(chip_npys)):
            print(f"[match] {i}/{len(chip_npys)}")

    # Write matches_topk.json
    out_js = os.path.join(sm_dir, "matches_topk.json")
    with open(out_js, "w") as f:
        json.dump(
            {
                "submap": sm_id,
                "topk": int(args.topk),
                "exclude_self": bool(args.exclude_self),
                "mode": args.mode,
                "max_edge": int(args.max_edge),
                "min_chip_valid_frac": float(args.min_chip_valid_frac),
                "min_chip_valid_count": int(args.min_chip_valid_count),
                "overfetch": int(args.overfetch),
                "matches": matches,
            },
            f,
            indent=2,
        )
    print(f"[ok] wrote {out_js}")

    # Optional background-safe heatmap
    if args.vis:
        counts = np.zeros(nx * ny, np.float32)
        for m in matches:
            for tid in m["top_ids"]:
                if 0 <= tid < counts.size:
                    counts[tid] += 1.0

        if counts.max() > 0:
            hm = (counts / (counts.max() + 1e-12)).reshape(ny, nx)
        else:
            hm = counts.reshape(ny, nx)

        # Keep background white if valid_ids is available
        if valid_ids is not None:
            mask = np.zeros_like(counts, dtype=bool)
            for tid in valid_ids:
                if 0 <= tid < mask.size:
                    mask[tid] = True
            flat = hm.reshape(-1)
            flat[~mask] = 0.0
            hm = flat.reshape(ny, nx)

        mq_path = os.path.join(args.root, "mosaic_quicklook.png")
        mq = cv2.imread(mq_path, cv2.IMREAD_COLOR)
        if mq is None:
            tile_px = 512
            mq = np.full((ny * tile_px, nx * tile_px, 3), 255, np.uint8)

        H, W = mq.shape[:2]
        hm_u8 = cv2.resize((hm * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        color[hm_u8 == 0] = (255, 255, 255)  # white background where no hits
        overlay = cv2.addWeighted(mq, 0.65, color, 0.35, 0)
        out_png = os.path.join(sm_dir, "covis_heatmap.png")
        cv2.imwrite(out_png, overlay)
        print(f"[ok] wrote {out_png}")

    # Debug summary
    print(
        f"[debug] skipped_too_sparse={n_skipped_too_sparse}, "
        f"empties_after_validRelax={n_empty_after_valid}, "
        f"empties_after_selfRelax={n_empty_after_self}"
    )


if __name__ == "__main__":
    try:
        main()
        print("[done] matching complete.")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
