#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, glob, shutil
from pathlib import Path

def _read_best_gallery_submap(root, sm):
    p = os.path.join(root, "submaps", sm, "faiss_votes_by_submap.json")
    j = json.load(open(p, "r"))
    ranking = j.get("ranking", [])
    if not ranking:
        return None
    return ranking[0][0]  # submap id string

def _tiles_of_submap(root, sm):
    tj = os.path.join(root, "submaps", sm, "tiles.json")
    j = json.load(open(tj, "r"))
    return list(map(int, j.get("overlapped_tile_ids", [])))

def main():
    ap = argparse.ArgumentParser("Prepare AnyLoc IO for submap-vs-submap (queries=chips, gallery=tiles)")
    ap.add_argument("--root", required=True, help="outputs/run")
    ap.add_argument("--submap", required=True, help="registering submap, e.g., sm_00001")
    ap.add_argument("--gallery-submap", default="", help="(optional) force gallery submap; default: take best from faiss_votes_by_submap.json")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlink")
    args = ap.parse_args()

    root = args.root
    sm_q = args.submap
    sm_g = args.gallery_submap or _read_best_gallery_submap(root, sm_q)
    if not sm_g:
        raise SystemExit("No gallery submap determined. Run submap_faiss_vote.py first or pass --gallery-submap.")

    out_root = os.path.join(root, "anyloc", f"{sm_q}_vs_{sm_g}")
    qdir = os.path.join(out_root, "queries")
    gdir = os.path.join(out_root, "gallery")
    Path(qdir).mkdir(parents=True, exist_ok=True)
    Path(gdir).mkdir(parents=True, exist_ok=True)

    # queries = CHIP PNGs of registering submap
    chips_dir = os.path.join(root, "submaps", sm_q, "chips")
    q_pngs = sorted(glob.glob(os.path.join(chips_dir, "*.png")))
    qlist = []
    for p in q_pngs:
        dst = os.path.join(qdir, os.path.basename(p))
        if os.path.exists(dst): os.remove(dst)
        if args.copy: shutil.copy2(p, dst)
        else: os.symlink(os.path.relpath(p, qdir), dst)
        qlist.append(os.path.basename(dst))

    # gallery = TILE PNGs of best submap’s tiles
    tids = _tiles_of_submap(root, sm_g)
    g_pngs = []
    for tid in tids:
        src = os.path.join(root, "tiles", f"tile_{tid:05d}.png")
        if os.path.isfile(src):
            dst = os.path.join(gdir, f"tile_{tid:05d}.png")
            if os.path.exists(dst): os.remove(dst)
            if args.copy: shutil.copy2(src, dst)
            else: os.symlink(os.path.relpath(src, gdir), dst)
            g_pngs.append(os.path.basename(dst))

    # write manifest lists
    with open(os.path.join(out_root, "queries.txt"), "w") as f:
        f.write("\n".join(qlist) + ("\n" if qlist else ""))
    with open(os.path.join(out_root, "gallery.txt"), "w") as f:
        f.write("\n".join(g_pngs) + ("\n" if g_pngs else ""))

    print(f"[anyloc-io] queries: {len(qlist)}  gallery: {len(g_pngs)}")
    print(f"[anyloc-io] folder: {out_root}")
    print("Now run AnyLoc with these two folders/files (see their README).")

if __name__ == "__main__":
    main()
