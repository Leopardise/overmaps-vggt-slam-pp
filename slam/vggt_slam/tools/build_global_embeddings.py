#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob, argparse
from pathlib import Path
import numpy as np
import cv2
import faiss

from vggt_slam.covis.embed_from_dem import DinoGrayEmbedder

def main():
    ap = argparse.ArgumentParser("Embed global tiles (gray) and build FAISS")
    ap.add_argument("--root", required=True, help="outputs/run")
    ap.add_argument("--dino-model", default="facebook/dinov2-base")
    ap.add_argument("--index", default="hnsw", choices=["hnsw","flatip"])
    ap.add_argument("--hnsw-M", type=int, default=32)
    ap.add_argument("--hnsw-efC", type=int, default=200)
    args = ap.parse_args()

    tiles_dir = os.path.join(args.root, "tiles")
    out_dir   = os.path.join(args.root, "global_embeddings")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    embedder = DinoGrayEmbedder(model_name=args.dino_model)

    ids, vecs = [], []
    for p in sorted(glob.glob(os.path.join(tiles_dir, "tile_*.png"))):
        tid = int(os.path.basename(p)[5:10])
        u8 = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if u8 is None: continue
        v = embedder.embed_u8gray(u8).astype(np.float32)
        vecs.append(v); ids.append(tid)

    if not vecs:
        raise RuntimeError("No tile PNGs found to embed.")

    X = np.stack(vecs, 0)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    d = X.shape[1]

    if args.index == "flatip":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexHNSWFlat(d, args.hnsw_M)
        index.hnsw.efConstruction = args.hnsw_efC
    index.add(X)

    faiss.write_index(index, os.path.join(out_dir, f"faiss_{args.index}.index"))
    with open(os.path.join(out_dir, "idmap.json"), "w") as f:
        json.dump({"ids": ids}, f, indent=2)
    print(f"[faiss] wrote {args.index} with N={len(ids)} dim={d} in {out_dir}")

if __name__ == "__main__":
    main()
