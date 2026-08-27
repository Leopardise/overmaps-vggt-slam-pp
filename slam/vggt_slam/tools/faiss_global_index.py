#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, json, argparse, glob
from pathlib import Path
import numpy as np

def _load_manifest(man_p):
    """
    Try a few manifest layouts and return [(tile_id, embed_path), ...]
    Supported:
      - {"vectors":[{"tile":"tile_00012.npy","embed":".../tile_00012.embed.npy"}, ...]}
      - {"embeddings":[{"tile_id":12,"path":".../tile_00012.embed.npy"}, ...]}
      - {"tiles":[{"id":12,"path":".../tile_00012.embed.npy"} ...]}
    """
    if not os.path.isfile(man_p):
        return None
    try:
        j = json.load(open(man_p, "r"))
    except Exception:
        return None

    items = []

    # New format written by embed_global_tiles.py
    if isinstance(j, dict) and "vectors" in j and isinstance(j["vectors"], list):
        for it in j["vectors"]:
            # Derive tid from file name (robust) or optional explicit id
            tid = None
            if "tile" in it:
                m = re.search(r"(\d+)", str(it["tile"]))
                if m: tid = int(m.group(1))
            if tid is None and "id" in it:
                tid = int(it["id"])
            pth = it.get("embed", "") or it.get("path", "")
            if tid is not None and pth and os.path.isfile(pth):
                items.append((tid, pth))
        if items:
            return items

    # Older alternates
    if isinstance(j, dict) and "embeddings" in j and isinstance(j["embeddings"], list):
        for it in j["embeddings"]:
            tid = int(it.get("tile_id", -1))
            pth = it.get("path", "")
            if tid >= 0 and os.path.isfile(pth):
                items.append((tid, pth))
        if items:
            return items

    if isinstance(j, dict) and "tiles" in j and isinstance(j["tiles"], list):
        for it in j["tiles"]:
            tid = int(it.get("id", -1))
            pth = it.get("path", it.get("embed_path", ""))
            if tid >= 0 and pth and os.path.isfile(pth):
                items.append((tid, pth))
        if items:
            return items

    return None

def _scan_embeddings(root):
    """Fallback scan: prefer global_embeddings/*.npy, then tiles/tile_*.embed.npy."""
    cand = []
    geb = os.path.join(root, "global_embeddings")
    if os.path.isdir(geb):
        cand.extend(sorted(glob.glob(os.path.join(geb, "*.npy"))))
    tiles = os.path.join(root, "tiles")
    cand.extend(sorted(glob.glob(os.path.join(tiles, "tile_*.embed.npy"))))

    items = []
    rx = re.compile(r"tile_(\d+).*\.embed\.npy$")
    for p in cand:
        m = rx.search(os.path.basename(p))
        if not m:
            continue
        tid = int(m.group(1))
        items.append((tid, p))

    # De-dup by tid, prefer global_embeddings path if present
    seen = {}
    for tid, p in items:
        if tid not in seen:
            seen[tid] = p
        else:
            # prefer the one under global_embeddings if available
            if "/global_embeddings/" in p and "/global_embeddings/" not in seen[tid]:
                seen[tid] = p
    return [(tid, seen[tid]) for tid in sorted(seen.keys())]

def _load_vectors(pairs):
    ids, vecs, paths = [], [], []
    for tid, p in pairs:
        try:
            v = np.load(p)
        except Exception:
            continue
        v = np.asarray(v)
        if v.ndim != 1:
            continue  # skip anything that isn't a 1-D embedding
        ids.append(int(tid))
        vecs.append(v.astype(np.float32))
        paths.append(p)
    if not vecs:
        raise RuntimeError("No 1-D embedding vectors found.")
    X = np.stack(vecs, axis=0).astype(np.float32)
    # Cosine similarity via normalized inner product
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / n
    return np.array(ids, dtype=np.int64), X, paths

def build_index(ids, X, index_type="hnsw", hnsw_M=32, hnsw_efC=200):
    import faiss
    d = X.shape[1]
    if index_type == "flatip":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexHNSWFlat(d, int(hnsw_M))
        index.hnsw.efConstruction = int(hnsw_efC)
    index.add(X)
    return index

def main():
    ap = argparse.ArgumentParser("Build FAISS index over global tile embeddings (windowed-embedding compatible)")
    ap.add_argument("--root", required=True, help="e.g., outputs/run")
    ap.add_argument("--index", default="hnsw", choices=["hnsw","flatip"])
    ap.add_argument("--hnsw-M", type=int, default=32)
    ap.add_argument("--hnsw-efC", type=int, default=200)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, "global_embeddings")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Prefer manifest from global_embeddings/index.json
    man_from = os.path.join(out_dir, "index.json")
    pairs = _load_manifest(man_from)

    # 2) Fallback scan
    if not pairs:
        pairs = _scan_embeddings(root)

    if not pairs:
        raise RuntimeError("Found no embeddings. Expected *.npy in global_embeddings/ or tiles/tile_*.embed.npy")

    ids, X, paths = _load_vectors(pairs)

    # sort by id
    order = np.argsort(ids)
    ids = ids[order]; X = X[order]; paths = [paths[i] for i in order]

    index = build_index(ids, X, args.index, args.hnsw_M, args.hnsw_efC)

    import faiss
    idx_path = os.path.join(out_dir, f"faiss_{args.index}.index")
    faiss.write_index(index, idx_path)
    idmap_path = os.path.join(out_dir, "idmap.json")
    with open(idmap_path, "w") as f:
        json.dump({"ids": ids.tolist(), "paths": paths}, f, indent=2)

    print(f"[faiss] wrote {idx_path}  (N={len(ids)}, dim={X.shape[1]})")
    print(f"[faiss] idmap: {idmap_path}")

if __name__ == "__main__":
    main()
