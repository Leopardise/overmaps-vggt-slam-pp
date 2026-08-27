#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob
from pathlib import Path
import numpy as np
import torch, cv2

from vggt_slam.covis.dem_u8 import dem_to_uint8_gray

def load_dino(device="cuda"):
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
    except Exception:
        # Expect local install if no internet; change to your local import if needed
        raise RuntimeError("DINOv2 not found via torch.hub; install facebookresearch/dinov2 or adjust loader.")
    model.eval().to(device)
    return model

@torch.no_grad()
def dino_embed(model, imgs_u8, device="cuda"):
    import torch.nn.functional as F
    arr = []
    for u8 in imgs_u8:
        if u8.ndim == 2:
            u8 = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        x = torch.from_numpy(u8).permute(2,0,1).float()/255.0
        x = F.interpolate(x[None], size=(518,518), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485,0.456,0.406], device=x.device)[None,:,None,None]
        std  = torch.tensor([0.229,0.224,0.225], device=x.device)[None,:,None,None]
        x = (x.to(device)-mean)/std
        f = model.forward_features(x)["x_norm_clstoken"].squeeze(0).float().cpu().numpy()
        arr.append(f.astype(np.float32))
    return np.stack(arr,0)

def main(out_root: str):
    tiles_dir = os.path.join(out_root, "tiles")
    emb_dir   = os.path.join(out_root, "global_embeddings")
    Path(emb_dir).mkdir(parents=True, exist_ok=True)

    index = json.load(open(os.path.join(out_root, "index.json"), "r"))
    clip_lo, clip_hi = index.get("clip_lo", 1.0), index.get("clip_hi", 99.0)

    paths = sorted(glob.glob(os.path.join(tiles_dir, "tile_*.npy")))
    if not paths:
        raise RuntimeError("No tiles found; run the global renderer first.")

    imgs, ids = [], []
    for p in paths:
        dem = np.load(p).astype(np.float32)
        u8  = dem_to_uint8_gray(dem, clip_lo=clip_lo, clip_hi=clip_hi)
        imgs.append(u8); ids.append(int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_dino(device)
    feats  = dino_embed(model, imgs, device)

    man = {"tiles": []}
    for tid, vec in zip(ids, feats):
        np.save(os.path.join(emb_dir, f"emb_tile_{tid:05d}.npy"), vec.astype(np.float32))
        man["tiles"].append({"id": int(tid), "path": f"emb_tile_{tid:05d}.npy"})
    json.dump(man, open(os.path.join(emb_dir, "tiles_manifest.json"), "w"), indent=2)
    print(f"[global-embed] wrote {len(ids)} embeddings → {emb_dir}")

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    args = ap.parse_args()
    main(args.out_root)
