#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob, numpy as np, faiss

def load_index(emb_dir: str):
    index = faiss.read_index(os.path.join(emb_dir, "faiss_hnsw.index"))
    idmap = json.load(open(os.path.join(emb_dir, "idmap.json"),"r"))["ids"]
    return index, np.asarray(idmap, np.int32)

def main(out_root: str, sm_dir: str, topk: int = 20):
    emb_dir = os.path.join(out_root, "global_embeddings")
    index, idmap = load_index(emb_dir)

    vecs = []; chip_ids = []
    for p in sorted(glob.glob(os.path.join(sm_dir, "chips", "chip_*_emb.npy"))):
        chip_ids.append(int(os.path.basename(p).split("_")[1].split(".")[0]))
        v = np.load(p).astype(np.float32)
        faiss.normalize_L2(v.reshape(1,-1))
        vecs.append(v)
    if not vecs:
        print(f"[match] no chip embeddings in {sm_dir}")
        return

    X = np.stack(vecs,0).astype(np.float32)
    D, I = index.search(X, topk)  # cosine (IP on L2-normalized)
    out = []
    for cid, d, i in zip(chip_ids, D, I):
        tiles = [{"tile_id": int(idmap[j]), "score": float(s)} for j, s in zip(i, d)]
        out.append({"chip_id": int(cid), "topk": tiles})
    json.dump(out, open(os.path.join(sm_dir, "matches_topk.json"), "w"), indent=2)
    print(f"[match] wrote matches_topk.json for {sm_dir}")

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--sm_dir", type=str, required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    main(args.out_root, args.sm_dir, args.topk)
