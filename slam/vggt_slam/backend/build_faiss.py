#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob
import numpy as np
import faiss

def main(emb_dir: str, hnsw_m: int = 32, efC: int = 200):
    vecs, ids = [], []
    for p in sorted(glob.glob(os.path.join(emb_dir, "emb_tile_*.npy"))):
        ids.append(int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
        vecs.append(np.load(p).astype(np.float32))
    if not vecs:
        raise RuntimeError("No embeddings found. Run embed_global_tiles.py first.")
    X = np.stack(vecs, 0).astype(np.float32)
    faiss.normalize_L2(X)
    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efC
    index.add(X)
    faiss.write_index(index, os.path.join(emb_dir, "faiss_hnsw.index"))
    json.dump({"ids": ids}, open(os.path.join(emb_dir, "idmap.json"), "w"), indent=2)
    print(f"[faiss] HNSW built with {len(ids)} vectors, dim={d}")

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=str, required=True)
    ap.add_argument("--hnsw_m", type=int, default=32)
    ap.add_argument("--efC", type=int, default=200)
    args = ap.parse_args()
    main(args.emb_dir, args.hnsw_m, args.efC)
