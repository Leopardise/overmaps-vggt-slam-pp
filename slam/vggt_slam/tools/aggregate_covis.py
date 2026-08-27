#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np
import cv2

def load_idmap(emb_dir):
    im = json.load(open(os.path.join(emb_dir, "idmap.json"), "r"))["ids"]
    # for FLAT/HNSW we don’t need reverse lookup; we aggregate on tile_id directly
    return set(int(i) for i in im)

def main():
    ap = argparse.ArgumentParser("Aggregate submap chip→tile matches into a covis heatmap")
    ap.add_argument("--root", required=True, help="outputs/run")
    ap.add_argument("--submap", required=True, help="e.g. sm_00012 or 'latest'")
    ap.add_argument("--weight", default="votes", choices=["votes","sim"], help="accumulate votes or similarity")
    ap.add_argument("--out", default="", help="override output png/json path prefix")
    args = ap.parse_args()

    # choose submap
    sm_root = os.path.join(args.root, "submaps")
    if args.submap == "latest":
        subs = [d for d in os.listdir(sm_root) if d.startswith("sm_")]
        if not subs:
            raise SystemExit("No submaps found.")
        submap = sorted(subs)[-1]
    else:
        submap = args.submap
    sm_dir = os.path.join(sm_root, submap)

    # load global frame / grid
    idx = json.load(open(os.path.join(args.root, "index.json"), "r"))
    nx, ny = int(idx["nx"]), int(idx["ny"])

    # matches file (output of submap_chip_embed_match.py)
    mfile = os.path.join(sm_dir, "matches_topk.json")
    if not os.path.isfile(mfile):
        raise SystemExit(f"Not found: {mfile}. Run submap_chip_embed_match.py first.")

    M = json.load(open(mfile, "r"))
    # expected format: list per chip → {"chip_id": "...", "topk":[{"tile_id": int, "sim": float}, ...]}
    H = np.zeros((ny, nx), np.float32)
    for chip in M.get("chips", M):  # tolerate {"chips":[...]} or a plain list
        topk = chip.get("topk", [])
        if args.weight == "votes":
            for it in topk:
                tid = int(it["tile_id"])
                ty, tx = divmod(tid, nx)
                H[ty, tx] += 1.0
        else:
            for it in topk:
                tid = int(it["tile_id"])
                sim = float(it.get("sim", 0.0))
                ty, tx = divmod(tid, nx)
                H[ty, tx] += max(sim, 0.0)

    # normalize to [0,1]
    if H.max() > 0:
        Hn = H / H.max()
    else:
        Hn = H

    # resize to mosaic resolution and overlay
    mosaic_p = os.path.join(args.root, "mosaic_quicklook.png")
    mos = cv2.imread(mosaic_p)
    if mos is None:
        raise SystemExit(f"Missing {mosaic_p}")

    tile_px = int(idx["grid_size_px"])
    ds      = max(1, tile_px // 512)
    cell = tile_px // ds
    heat = cv2.resize(Hn, (nx*cell, ny*cell), interpolation=cv2.INTER_NEAREST)

    heat_color = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)  # 0..255
    overlay = mos.copy()
    alpha = 0.45
    mask = (heat > 0).astype(np.float32)[..., None]
    overlay = (overlay * (1 - alpha*mask) + heat_color * (alpha*mask)).astype(np.uint8)

    base = args.out or os.path.join(sm_dir, "covis")
    cv2.imwrite(base + "_heatmap.png", overlay)
    json.dump({"nx": nx, "ny": ny, "grid": H.tolist()}, open(base + ".json", "w"))
    print(f"[covis] wrote {base}_heatmap.png and {base}.json")

if __name__ == "__main__":
    main()
