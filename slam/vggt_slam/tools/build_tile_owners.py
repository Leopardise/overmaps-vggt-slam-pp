#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan outputs/<RUN>/submaps/*/tiles.json and build:
  - submap_index/tile_owners.json  : { tile_id (str) -> [submap_id, ...] }
  - submap_index/submap_tiles.json : { submap_id     -> [tile_id, ...] }

Handles overlaps (a tile can be owned by multiple submaps).
"""

import os, json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser("Build tile→submap ownership index from submap tiles.json files")
    ap.add_argument("--root", required=True, help="e.g., outputs/05")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    submaps_dir = os.path.join(root, "submaps")
    out_dir = os.path.join(root, "submap_index")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(submaps_dir):
        raise SystemExit(f"No submaps/ directory at {submaps_dir}")

    tile_owners = {}     # str(tile_id) -> list[submap_id]
    submap_tiles = {}    # submap_id -> list[int]

    n_missing = 0
    n_submaps = 0
    for name in sorted(os.listdir(submaps_dir)):
        if not name.startswith("sm_"):
            continue
        sm_dir = os.path.join(submaps_dir, name)
        if not os.path.isdir(sm_dir):
            continue
        n_submaps += 1
        tj = os.path.join(sm_dir, "tiles.json")
        if not os.path.isfile(tj):
            n_missing += 1
            continue
        try:
            data = json.load(open(tj, "r"))
            tids = data.get("overlapped_tile_ids", [])
            if not isinstance(tids, list):
                tids = []
        except Exception:
            tids = []
        submap_tiles[name] = sorted(int(t) for t in tids)
        for t in submap_tiles[name]:
            key = str(int(t))
            tile_owners.setdefault(key, [])
            if name not in tile_owners[key]:
                tile_owners[key].append(name)

    # write outputs
    owners_p = os.path.join(out_dir, "tile_owners.json")
    subs_p   = os.path.join(out_dir, "submap_tiles.json")

    if (not args.overwrite) and os.path.exists(owners_p):
        print(f"[info] {owners_p} exists; use --overwrite to rebuild.")
    else:
        with open(owners_p, "w") as f:
            json.dump({"tile_owners": tile_owners}, f, indent=2)
        print(f"[ok] wrote {owners_p} (tiles indexed: {len(tile_owners)})")

    if (not args.overwrite) and os.path.exists(subs_p):
        print(f"[info] {subs_p} exists; use --overwrite to rebuild.")
    else:
        with open(subs_p, "w") as f:
            json.dump({"submap_tiles": submap_tiles}, f, indent=2)
        print(f"[ok] wrote {subs_p} (submaps indexed: {len(submap_tiles)})")

    if n_missing > 0:
        print(f"[warn] {n_missing}/{n_submaps} submaps had no tiles.json (were they chipped yet?)")

if __name__ == "__main__":
    main()
