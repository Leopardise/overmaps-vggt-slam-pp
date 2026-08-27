#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On-the-fly VLAD over DINOv2 patches (AnyLoc-style), memory-safe.

- Streams patch descriptors per image on GPU (DINOv2 ViT-14).
- Fits MiniBatchKMeans on a sampled pool of patches (CPU).
- Aggregates VLAD per image in chunks (GPU-friendly), intra-normalizes, L2-normalizes.
- Performs cosine (IP) retrieval and writes CSV.

Inputs:
  --queries   outputs/run/anyloc_io/<SUB>/queries.txt
  --database  outputs/run/anyloc_io/<SUB>/database.txt
Outputs:
  --out       outputs/run/anyloc_io/<SUB>/matches_anyloc.csv

CSV columns:
  query_path,db_path,score
"""

import os, sys, csv, argparse, random
import numpy as np
import torch, cv2
from sklearn.cluster import MiniBatchKMeans

# ------------------ DINO helpers ------------------

def map_model_name(name: str) -> str:
    n = (name or "").lower().strip()
    if n in ["facebook/dinov2-base","dinov2_vitb14","dinov2-base","vitb14","vitb/14","b14"]:
        return "dinov2_vitb14"
    if n in ["facebook/dinov2-large","dinov2_vitl14","dinov2-large","vitl14","vitl/14","l14"]:
        return "dinov2_vitl14"
    if n in ["facebook/dinov2-giant","dinov2_vitg14","dinov2-giant","vitg14","vitg/14","g14"]:
        return "dinov2_vitg14"
    return "dinov2_vitb14"

def load_dino(model_name: str, device: str):
    m = map_model_name(model_name)
    print(f"[dino] {m} on {device}")
    mdl = torch.hub.load('facebookresearch/dinov2', m).eval().to(device)
    patch = 14 if "14" in m else 16
    return mdl, patch

def prep_image(path: str, mode: str, max_edge: int, patch: int) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    # normalize channels
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, axis=2)
    elif im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    H, W = im.shape[:2]
    # resize while keeping aspect
    if mode == "resize" and max(H, W) > max_edge:
        if H >= W:
            newH = max_edge; newW = int(round(W * (max_edge / H)))
        else:
            newW = max_edge; newH = int(round(H * (max_edge / W)))
        im = cv2.resize(im, (newW, newH), interpolation=cv2.INTER_AREA)
        H, W = im.shape[:2]
    # center-crop/pad to multiples of patch
    Hc = max(patch, (H // patch) * patch)
    Wc = max(patch, (W // patch) * patch)
    ph, pw = max(0, Hc - H), max(0, Wc - W)
    if ph or pw:
        im = cv2.copyMakeBorder(im, ph//2, ph-ph//2, pw//2, pw-pw//2, cv2.BORDER_REFLECT_101)
        H, W = im.shape[:2]
    y0, x0 = (H - Hc)//2, (W - Wc)//2
    im = im[y0:y0+Hc, x0:x0+Wc]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # contiguous, positive strides
    return im

@torch.no_grad()
def dino_patch_desc(model, rgb_uint8: np.ndarray, device: str) -> torch.Tensor:
    """Return [K,D] float32 patch descriptors using DINOv2 last-block tokens."""
    x = torch.from_numpy(rgb_uint8).permute(2,0,1).unsqueeze(0).float() / 255.0
    x = x.to(device, non_blocking=True)
    out = model.forward_features(x)
    # Use the penultimate token map if available; fallback to final tokens
    # DINOv2 forward_features doesn't expose per-patch easily; use 'x_norm_patchtokens' if present
    tokens = out.get("x_norm_patchtokens", None)
    if tokens is None:
        # fall back: take last hidden states then drop CLS
        # Some dinov2 builds return 'x_norm' of shape [N,1+K,D]
        xnorm = out.get("x_norm", None)
        if xnorm is None:
            raise RuntimeError("DINO forward_features missing patch tokens")
        tokens = xnorm[:, 1:, :]  # drop CLS
    desc = tokens.squeeze(0).float()  # [K,D]
    return desc

# ------------------ VLAD core ------------------

def fit_vlad_centers_stream(
    model, device, paths, *,
    clusters=64, mode="resize", max_edge=1024, patch=14,
    max_images_seed=1000, per_image_patch_sample=256, total_patch_cap=200000, seed=42
) -> np.ndarray:
    """
    Stream images → sample patch desc → MiniBatchKMeans → centers [C,D] (float32).
    """
    random.seed(seed); np.random.seed(seed)
    print(f"[kmeans] sampling up to {total_patch_cap} patches "
          f"(<= {per_image_patch_sample}/img, <= {max_images_seed} imgs), C={clusters}")

    pool = []
    taken = 0
    for i, p in enumerate(paths[:max_images_seed], 1):
        try:
            rgb = prep_image(p, mode, max_edge, patch)
            desc = dino_patch_desc(model, rgb, device).cpu().numpy()  # [K,D]
            K = desc.shape[0]
            if K <= per_image_patch_sample:
                sel = desc
            else:
                idx = np.random.choice(K, size=per_image_patch_sample, replace=False)
                sel = desc[idx]
            pool.append(sel)
            taken += sel.shape[0]
        except Exception as e:
            print(f"[warn] seed desc fail for {p}: {e}")
        if i % 50 == 0 or i == min(len(paths), max_images_seed):
            print(f"[kmeans] seed {i}/{min(len(paths), max_images_seed)} imgs, patches so far={taken}")
        if taken >= total_patch_cap:
            break

    if not pool:
        raise RuntimeError("No seed descriptors collected for KMeans.")
    X = np.concatenate(pool, axis=0)
    if X.shape[0] > total_patch_cap:
        idx = np.random.choice(X.shape[0], size=total_patch_cap, replace=False)
        X = X[idx]
    print(f"[kmeans] fitting MiniBatchKMeans on {X.shape[0]} patches, dim={X.shape[1]}")

    km = MiniBatchKMeans(
        n_clusters=clusters, batch_size=8192, max_iter=100,
        compute_labels=False, n_init=1, reassignment_ratio=0.01, random_state=seed
    )
    km.fit(X)
    C = km.cluster_centers_.astype(np.float32)  # [C,D]
    print("[kmeans] done.")
    return C

@torch.no_grad()
def vlad_embed_one(desc: torch.Tensor, centers: torch.Tensor, chunk=4096, intra=True) -> torch.Tensor:
    """
    desc:    [K,D] (device tensor)
    centers: [C,D] (device tensor)
    Returns: [C*D] VLAD vector (L2-normalized), torch.float32 on desc.device
    """
    K, D = desc.shape
    C = centers.shape[0]
    V = torch.zeros((C, D), dtype=desc.dtype, device=desc.device)

    # assign in chunks to avoid KxC blow-ups
    for s in range(0, K, chunk):
        e = min(K, s + chunk)
        x = desc[s:e]                                  # [B,D]
        # squared L2 distance: ||x||^2 + ||c||^2 - 2 x c^T
        x2 = (x*x).sum(dim=1, keepdim=True)            # [B,1]
        c2 = (centers*centers).sum(dim=1).unsqueeze(0) # [1,C]
        scores = x2 + c2 - 2.0 * (x @ centers.t())     # [B,C]
        a = scores.argmin(dim=1)                       # [B]
        # residuals and accumulate
        res = x - centers[a]                           # [B,D]
        V.index_add_(0, a, res)

    if intra:
        V = torch.nn.functional.normalize(V, dim=1)    # intra-norm per cluster
    v = V.reshape(-1)
    v = torch.nn.functional.normalize(v, dim=0)        # final L2
    return v

@torch.no_grad()
def vlad_embed_paths(paths, model, device, centers_np: np.ndarray,
                     mode="resize", max_edge=1024, patch=14) -> np.ndarray:
    centers = torch.from_numpy(centers_np).to(device)
    embs = []
    for i, p in enumerate(paths, 1):
        try:
            rgb = prep_image(p, mode, max_edge, patch)
            desc = dino_patch_desc(model, rgb, device)          # [K,D]
            v = vlad_embed_one(desc, centers)                   # [C*D]
            embs.append(v.float().cpu().numpy())
        except Exception as e:
            print(f"[warn] VLAD fail for {p}: {e}")
            embs.append(np.zeros(centers.numel()//centers.shape[0], np.float32))
        if i % 25 == 0 or i == len(paths):
            print(f"[vlad] {i}/{len(paths)}")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return np.stack(embs, 0).astype(np.float32)

# ------------------ retrieval ------------------

def ip_search(Q, D, topk=50):
    sims = Q @ D.T
    k = min(topk, sims.shape[1])
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    rows = []
    for i in range(Q.shape[0]):
        j = idx[i]
        s = sims[i, j]
        o = np.argsort(-s)
        rows.append((j[o], s[o]))
    return rows

def read_list(path):
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser("AnyLoc-style VLAD over DINOv2 (on-the-fly, memory-safe)")
    ap.add_argument("--queries",  required=True)
    ap.add_argument("--database", required=True)
    ap.add_argument("--out",      required=True)

    # DINO / resize
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--mode", choices=["crop","resize"], default="resize")
    ap.add_argument("--max-edge", type=int, default=1024)

    # VLAD centers (on-the-fly fitting controls)
    ap.add_argument("--clusters", type=int, default=64)
    ap.add_argument("--seed-max-images", type=int, default=1000)
    ap.add_argument("--seed-patch-sample", type=int, default=256)
    ap.add_argument("--seed-total-cap", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-centers", default="", help="optional path to save learned centers (.npy)")
    ap.add_argument("--load-centers", default="", help="optional path to reuse centers (.npy)")

    # retrieval
    ap.add_argument("--topk", type=int, default=50)

    args = ap.parse_args()

    q_paths = read_list(args.queries)
    d_paths = read_list(args.database)
    print(f"[io] {len(q_paths)} queries, {len(d_paths)} database")

    miss = [p for p in q_paths + d_paths if not os.path.exists(p)]
    if miss:
        print("[err] missing files, first few:", miss[:5]); sys.exit(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, patch = load_dino(args.backbone, device)

    # --- centers: load or fit ---
    if args.load_centers and os.path.isfile(args.load_centers):
        centers = np.load(args.load_centers).astype(np.float32)
        print(f"[centers] loaded {centers.shape} from {args.load_centers}")
    else:
        # fit from the *database* only (optional: include queries too)
        centers = fit_vlad_centers_stream(
            model, device, d_paths,
            clusters=args.clusters,
            mode=args.mode, max_edge=args.max_edge, patch=patch,
            max_images_seed=args.seed_max_images,
            per_image_patch_sample=args.seed_patch_sample,
            total_patch_cap=args.seed_total_cap,
            seed=args.seed
        )
        if args.save_centers:
            os.makedirs(os.path.dirname(args.save_centers), exist_ok=True)
            np.save(args.save_centers, centers)
            print(f"[centers] saved to {args.save_centers}")

    # --- embeddings ---
    Q = vlad_embed_paths(q_paths, model, device, centers, mode=args.mode, max_edge=args.max_edge, patch=patch)
    D = vlad_embed_paths(d_paths, model, device, centers, mode=args.mode, max_edge=args.max_edge, patch=patch)

    # --- retrieval ---
    rows = ip_search(Q, D, topk=args.topk)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_path","db_path","score"])
        for qi, (idxs, scrs) in enumerate(rows):
            qp = q_paths[qi]
            for dj, sc in zip(idxs.tolist(), scrs.tolist()):
                w.writerow([qp, d_paths[dj], f"{float(sc):.6f}"])
    print(f"[ok] wrote {args.out}")

if __name__ == "__main__":
    main()
