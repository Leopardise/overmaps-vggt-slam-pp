#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np
from pathlib import Path
import faiss
from collections import Counter

def load_faiss(root, index_kind="hnsw"):
    gdir = os.path.join(root, "global_embeddings")
    index = faiss.read_index(os.path.join(gdir, f"faiss_{index_kind}.index"))
    with open(os.path.join(gdir, "idmap.json"), "r") as f:
        ids = json.load(f)["ids"]
    return index, np.array(ids, dtype=np.int32)

def query_one_submap(root, submap_id, index, ids, topk=5):
    chips_dir = os.path.join(root, "submaps", submap_id, "embeddings")
    if not os.path.isdir(chips_dir):
        raise FileNotFoundError(f"no embeddings for {submap_id} at {chips_dir}")

    embs = []
    chips = []
    for f in sorted(os.listdir(chips_dir)):
        if f.endswith(".npy"):
            chips.append(int(f.replace(".npy","")))
            v = np.load(os.path.join(chips_dir, f)).astype(np.float32)
            n = np.linalg.norm(v) + 1e-12
            embs.append(v / n)
    if not embs:
        raise RuntimeError("no chip embeddings found")

    Q = np.stack(embs, 0)
    D, I = index.search(Q, topk)  # cosine since we normalized + IP index
    # voting by tile id
    votes = Counter()
    for row in I:
        for col in row:
            if col < 0: continue
            votes[int(ids[col])] += 1
    top = votes.most_common(20)

    return {
        "submap": submap_id,
        "num_chips": len(embs),
        "top_matches": [{"tile_id": tid, "votes": v} for tid, v in top]
    }

def main():
    ap = argparse.ArgumentParser("Match submap chips to global tiles via FAISS")
    ap.add_argument("--root", required=True)
    ap.add_argument("--submap", required=True, help="e.g., sm_00007")
    ap.add_argument("--index-kind", default="hnsw", choices=["hnsw","flatip"])
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    index, ids = load_faiss(args.root, args.index_kind)
    res = query_one_submap(args.root, args.submap, index, ids, topk=args.topk)

    out = os.path.join(args.root, "submaps", args.submap, "matches.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[match] wrote {out}")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
