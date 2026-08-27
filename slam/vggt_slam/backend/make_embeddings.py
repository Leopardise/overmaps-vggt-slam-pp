#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, numpy as np, cv2, torch

def load_dino(device="cuda"):
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
    except Exception:
        raise RuntimeError("DINOv2 not found via torch.hub; install facebookresearch/dinov2 or adjust loader.")
    model.eval().to(device)
    return model

@torch.no_grad()
def dino_embed(model, imgs_u8, device="cuda"):
    import torch.nn.functional as F
    arr = []
    for u8 in imgs_u8:
        if u8.ndim == 2: u8 = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        x = torch.from_numpy(u8).permute(2,0,1).float()/255.0
        x = F.interpolate(x[None], size=(518,518), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485,0.456,0.406], device=x.device)[None,:,None,None]
        std  = torch.tensor([0.229,0.224,0.225], device=x.device)[None,:,None,None]
        x = (x.to(device)-mean)/std
        f = model.forward_features(x)["x_norm_clstoken"].squeeze(0).float().cpu().numpy()
        arr.append(f.astype(np.float32))
    return np.stack(arr,0)

def main(chips_dir: str):
    imgs, ids = [], []
    for p in sorted(glob.glob(os.path.join(chips_dir, "chip_*_u8.png"))):
        u8 = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        imgs.append(u8)
        ids.append(int(os.path.basename(p).split("_")[1].split(".")[0]))
    if not imgs:
        print(f"[embed] no chips in {chips_dir}")
        return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_dino(dev)
    X = dino_embed(model, imgs, dev)

    for tid, vec in zip(ids, X):
        np.save(os.path.join(chips_dir, f"chip_{tid:05d}_emb.npy"), vec.astype(np.float32))
    print(f"[embed] wrote {len(ids)} chip embeddings → {chips_dir}")
    return len(ids)

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--chips_dir", type=str, required=True)
    args = ap.parse_args()
    main(args.chips_dir)
