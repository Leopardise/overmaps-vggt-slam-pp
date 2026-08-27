#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay submap ownership on mosaic_quicklook.png with a clean legend.

Fixes:
- Cell size derived from mosaic dims (W/nx, H/ny), no heuristics.
- Deterministic owner selection (vote, then submap id).
- Uses submap_index/tile_owners.json if present; else scans submaps/*/tiles.json.

Outputs:
  <root>/mosaic_submap_colors.png
  <root>/mosaic_submap_colors_with_grid.png   (if --grid)
  <root>/submap_color_legend.png
"""

import os, json, glob, argparse, math, hashlib, colorsys
from collections import defaultdict
import numpy as np
import cv2

def _read_json(p):
    with open(p, "r") as f:
        return json.load(f)

def load_index(root):
    idxp = os.path.join(root, "index.json")
    if not os.path.isfile(idxp):
        raise FileNotFoundError(f"index.json not found in {root}")
    idx = _read_json(idxp)

    def _get_int(d, *keys, default=None):
        for k in keys:
            if k in d:
                try: return int(d[k])
                except Exception: pass
        return default

    nx = _get_int(idx, "nx", "num_tiles_x", "grid_nx")
    ny = _get_int(idx, "ny", "num_tiles_y", "grid_ny")
    if nx is None or ny is None:
        raise KeyError("nx/ny missing in index.json")

    return idx, int(nx), int(ny)

def load_mosaic(root):
    cands = [
        os.path.join(root, "mosaic_quicklook.png"),
        os.path.join(root, "mosaic_quicklook_with_odo.png"),
    ]
    for m in cands:
        im = cv2.imread(m, cv2.IMREAD_COLOR)
        if im is not None: return im, m
    raise FileNotFoundError("mosaic_quicklook*.png not found")

def stable_color_for_submap(sm_id: str):
    h = int(hashlib.sha1(sm_id.encode("utf-8")).hexdigest(), 16)
    hue = (h % 360) / 360.0
    sat = 0.70
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(round(255*b)), int(round(255*g)), int(round(255*r)))  # BGR

def load_tile_owners(root):
    """Return dict: tid(int) -> list[(submap_id, vote_float)]"""
    owners_path = os.path.join(root, "submap_index", "tile_owners.json")
    if os.path.isfile(owners_path):
        j = _read_json(owners_path)
        out = defaultdict(list)
        for k, lst in j.get("tile_owners", {}).items():
            tid = int(k)
            for sm in lst:
                out[tid].append((sm, 1.0))  # stored file is membership; weight=1
        return out

    # Fallback: scan submaps/*/tiles.json
    out = defaultdict(list)
    for sm_tj in sorted(glob.glob(os.path.join(root, "submaps", "sm_*", "tiles.json"))):
        sm = os.path.basename(os.path.dirname(sm_tj))
        try:
            data = _read_json(sm_tj)
        except Exception:
            continue
        votes = data.get("tile_votes")
        if isinstance(votes, dict):
            for k, v in votes.items():
                try:
                    out[int(k)].append((sm, float(v)))
                except Exception:
                    pass
        for tid in data.get("overlapped_tile_ids", []):
            try:
                out[int(tid)].append((sm, 1.0))
            except Exception:
                pass
    return out

def pick_owner(owners_for_tile):
    """Pick owner deterministically by (vote desc, submap id asc)."""
    if not owners_for_tile:
        return None
    return sorted(owners_for_tile, key=lambda t: (-float(t[1]), t[0]))[0][0]

def draw_grid(img, nx, ny, cell_w, cell_h, color=(230,230,230), thick=1):
    H, W = img.shape[:2]
    for x in range(nx+1):
        X = min(W-1, int(round(x * cell_w)))
        cv2.line(img, (X,0), (X,H-1), color, thick, cv2.LINE_AA)
    for y in range(ny+1):
        Y = min(H-1, int(round(y * cell_h)))
        cv2.line(img, (0,Y), (W-1,Y), color, thick, cv2.LINE_AA)

def make_legend(submap_colors, cols=4, swatch=20, pad=10, font_scale=0.5, thickness=1):
    entries = list(sorted(submap_colors.items(), key=lambda kv: kv[0]))
    if not entries:
        return np.full((40, 200, 3), 255, np.uint8)

    rows = math.ceil(len(entries) / cols)
    text_h = 18
    cell_w = 210
    cell_h = max(swatch, text_h) + 8

    W = cols * cell_w + (cols+1)*pad
    H = rows * cell_h + (rows+1)*pad
    out = np.full((H, W, 3), 255, np.uint8)

    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(entries): break
            sm, color = entries[i]
            x0 = pad + c*cell_w
            y0 = pad + r*cell_h
            cv2.rectangle(out, (x0, y0), (x0+swatch, y0+swatch), color, -1, cv2.LINE_AA)
            cv2.putText(out, sm, (x0+swatch+8, y0+swatch-4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (40,40,40), thickness, cv2.LINE_AA)
            i += 1
    return out

def main():
    ap = argparse.ArgumentParser("Overlay submap ownership on mosaic quicklook + legend")
    ap.add_argument("--root", required=True, help="Global DEM run directory")
    ap.add_argument("--alpha", type=float, default=0.35, help="overlay strength (0..1)")
    ap.add_argument("--grid", action="store_true", help="also save gridlined version")
    ap.add_argument("--legend-cols", type=int, default=4)
    args = ap.parse_args()

    _, nx, ny = load_index(args.root)
    mosaic, _ = load_mosaic(args.root)
    H, W = mosaic.shape[:2]

    # Exact cell size from mosaic dims (no heuristics)
    cell_w = W / float(nx)
    cell_h = H / float(ny)

    owners = load_tile_owners(args.root)
    # collect all submaps that appear at least once
    submaps = sorted({sm for lst in owners.values() for (sm, _) in lst})
    submap_colors = {sm: stable_color_for_submap(sm) for sm in submaps}

    overlay = mosaic.copy()
    claimed = 0

    for tid, lst in owners.items():
        sm = pick_owner(lst)
        if sm is None:
            continue
        color = submap_colors.get(sm, (0,0,0))
        ty, tx = divmod(int(tid), nx)
        # rectangle in mosaic pixel coords
        x0 = int(round(tx * cell_w)); x1 = int(round((tx + 1) * cell_w))
        y0 = int(round(ty * cell_h)); y1 = int(round((ty + 1) * cell_h))
        x0 = max(0, min(W-1, x0)); x1 = max(x0+1, min(W, x1))
        y0 = max(0, min(H-1, y0)); y1 = max(y0+1, min(H, y1))
        cv2.rectangle(overlay, (x0, y0), (x1-1, y1-1), color, thickness=-1, lineType=cv2.LINE_AA)
        claimed += 1

    a = float(np.clip(args.alpha, 0.0, 1.0))
    colored = cv2.addWeighted(overlay, a, mosaic, 1.0 - a, 0)

    if args.grid:
        grid = colored.copy()
        draw_grid(grid, nx, ny, cell_w, cell_h, color=(240,240,240), thick=1)

    legend = make_legend(submap_colors, cols=args.legend_cols)

    out1 = os.path.join(args.root, "mosaic_submap_colors.png")
    cv2.imwrite(out1, colored)
    if args.grid:
        out2 = os.path.join(args.root, "mosaic_submap_colors_with_grid.png")
        cv2.imwrite(out2, grid)
    outl = os.path.join(args.root, "submap_color_legend.png")
    cv2.imwrite(outl, legend)

    print(f"[ok] colored mosaic:          {out1}")
    if args.grid:
        print(f"[ok] colored mosaic (grid):  {out2}")
    print(f"[ok] legend:                  {outl}")
    print(f"[info] tiles with owners:     {claimed} / {nx*ny}")
    print(f"[info] unique submaps:        {len(submaps)}")

if __name__ == "__main__":
    main()
