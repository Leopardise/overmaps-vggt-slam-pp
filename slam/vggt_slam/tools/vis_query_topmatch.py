#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize query submap vs its top-1 matched submap (from AnyLoc matches.csv).

- Rank-weight top-K patch matches per chip (weights K..1), exclude self.
- Aggregate by submap via submap_index/tile_owners.json.
- Pick top-2; visualize query and top-1 on mosaic_quicklook:
    * outline query tiles (color A)
    * outline top-1 tiles (color B)
    * draw a line between submap centroids
    * annotate IDs and scores + small legend

Usage:
python vggt_slam/tools/vis_query_topmatch.py \
  --root outputs/05 \
  --submap sm_00003 \
  --matches outputs/05/anyloc_io/sm_00003/matches.csv \
  --per-chip-topk 5 \
  --out outputs/05/anyloc_io/sm_00003/top1_overlay.png
"""

import os, re, csv, json, argparse, math
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import cv2


# ---------------- I/O helpers ----------------

def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _load_index(root: str):
    """Return (nx, ny, tile_px) with fallbacks; tolerate various schemas."""
    idxp = os.path.join(root, "index.json")
    idx = _read_json(idxp)

    def pick(d, keys, default=None, cast=int):
        for k in keys:
            if k in d:
                try:
                    return cast(d[k])
                except Exception:
                    pass
        return default

    nx = pick(idx, ["nx", "num_tiles_x", "grid_nx"])
    ny = pick(idx, ["ny", "num_tiles_y", "grid_ny"])
    tile_px = pick(idx, ["tile_px", "tile_size_px", "grid_size_px", "grid_px"], default=512)

    if nx is None or ny is None:
        raise KeyError(f"nx/ny missing in {idxp}")
    return int(nx), int(ny), int(tile_px)

def _load_mosaic(root: str):
    cands = [
        os.path.join(root, "mosaic_quicklook.png"),
        os.path.join(root, "mosaic_quicklook_with_odo.png"),
    ]
    for p in cands:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is not None:
            return im, p
    raise FileNotFoundError(f"Could not find mosaic quicklook at: {cands}")

def _load_tile_owners(root: str) -> Dict[int, List[str]]:
    """tile_id -> [sm_XXXXX, ...]"""
    p = os.path.join(root, "submap_index", "tile_owners.json")
    j = _read_json(p)
    own = j.get("tile_owners", j)  # support flat dict too
    out = {}
    for k, v in own.items():
        try:
            out[int(k)] = list(v)
        except Exception:
            pass
    return out

def _invert_owners(owners: Dict[int, List[str]]) -> Dict[str, Set[int]]:
    """sm -> set(tile_ids)"""
    sm2tiles: Dict[str, Set[int]] = defaultdict(set)
    for tid, sms in owners.items():
        for sm in sms:
            sm2tiles[sm].add(tid)
    return sm2tiles

def _tile_id_from_path(path: str):
    """Parse .../tile_00042.png (or .jpg/.npy) → 42"""
    m = re.search(r"tile_(\d+)\.(?:png|jpg|jpeg|npy)$", os.path.basename(path), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def _read_matches(matches_csv: str):
    """Return dict: query_path -> [(db_path, score)]"""
    groups = defaultdict(list)
    with open(matches_csv, "r") as f:
        rd = csv.DictReader(f)
        need = {"query_path", "db_path", "score"}
        if not need.issubset(rd.fieldnames or []):
            raise SystemExit(f"{matches_csv} must contain columns: {sorted(list(need))}")
        for row in rd:
            try:
                s = float(row["score"])
            except Exception:
                s = 0.0
            groups[row["query_path"]].append((row["db_path"], s))
    return groups


# ---------------- voting & ranking ----------------

def _rank_weight_votes(groups, owners, self_submap: str, per_chip_topk: int):
    """Return sorted list[(sm, score)], descending. Excludes self_submap."""
    submap_scores = defaultdict(float)
    for qp, lst in groups.items():
        lst.sort(key=lambda x: -x[1])
        top = lst[:per_chip_topk]
        K = len(top)
        for r, (dbp, _) in enumerate(top):
            w = float(K - r)  # K..1
            tid = _tile_id_from_path(dbp)
            if tid is None:
                continue
            for sm in owners.get(tid, []):
                if sm != self_submap:
                    submap_scores[sm] += w
    ranked = sorted(submap_scores.items(), key=lambda kv: -kv[1])
    return ranked  # [(sm, score), ...]


# ---------------- geometry helpers ----------------

def _grid_cell_size(mosaic: np.ndarray, nx: int, ny: int, tile_px: int):
    H, W = mosaic.shape[:2]
    # default downsample factor ds = tile_px // 512 if mosaic built that way
    ds_guess = max(1, tile_px // 512)
    cw = tile_px // ds_guess
    ch = tile_px // ds_guess
    # If mismatch, compute from mosaic dims
    if abs(nx * cw - W) > 3 or abs(ny * ch - H) > 3:
        cw = W / float(nx)
        ch = H / float(ny)
    return float(cw), float(ch)

def _tile_rect(tid: int, nx: int, cw: float, ch: float, W: int, H: int):
    ty, tx = divmod(int(tid), nx)
    x0 = int(round(tx * cw)); x1 = int(round((tx + 1) * cw))
    y0 = int(round(ty * ch)); y1 = int(round((ty + 1) * ch))
    # clamp
    x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W, x1))
    y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H, y1))
    return x0, y0, x1, y1

def _centroid_from_tiles(tids: Set[int], nx: int, cw: float, ch: float):
    if not tids:
        return None
    xs, ys, n = 0.0, 0.0, 0
    for tid in tids:
        ty, tx = divmod(int(tid), nx)
        # center of the cell
        cx = (tx + 0.5) * cw
        cy = (ty + 0.5) * ch
        xs += cx; ys += cy; n += 1
    return (xs / n, ys / n) if n else None


# ---------------- drawing ----------------

def _draw_outline_rects(img, tids: Set[int], nx: int, cw: float, ch: float, color, thick=2):
    H, W = img.shape[:2]
    for tid in tids:
        x0, y0, x1, y1 = _tile_rect(tid, nx, cw, ch, W, H)
        cv2.rectangle(img, (x0, y0), (x1-1, y1-1), color, thick, cv2.LINE_AA)

def _draw_label(img, text: str, org: Tuple[int,int], color=(255,255,255), bg=(30,30,30)):
    pad = 4
    ((tw, th), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - 2*pad), (x + tw + 2*pad, y + pad), bg, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + pad, y - pad), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def _draw_legend(panel_w=300, panel_h=80, a_color=(0,200,255), b_color=(255,80,180)):
    lg = np.full((panel_h, panel_w, 3), 255, np.uint8)
    # query
    cv2.rectangle(lg, (12, 18), (42, 48), a_color, -1, cv2.LINE_AA)
    cv2.putText(lg, "Query submap tiles", (52, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,40,40), 1, cv2.LINE_AA)
    # top1
    cv2.rectangle(lg, (12, 58), (42, 88), b_color, -1, cv2.LINE_AA)
    cv2.putText(lg, "Top-1 matched submap tiles", (52, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,40,40), 1, cv2.LINE_AA)
    return lg


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser("Visualize query vs top-1 matched submap on mosaic_quicklook.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--submap", required=True, help="query submap, e.g. sm_00003")
    ap.add_argument("--matches", required=True, help="AnyLoc matches.csv for the query submap")
    ap.add_argument("--per-chip-topk", type=int, default=5, help="rank weighting up to top-K per chip (K..1)")
    ap.add_argument("--out", default="", help="output PNG; defaults to $ROOT/anyloc_io/<submap>/top1_overlay.png")
    args = ap.parse_args()

    nx, ny, tile_px = _load_index(args.root)
    mosaic, mpath = _load_mosaic(args.root)
    H, W = mosaic.shape[:2]
    cw, ch = _grid_cell_size(mosaic, nx, ny, tile_px)

    owners = _load_tile_owners(args.root)           # tid -> [sm_...]
    sm2tiles = _invert_owners(owners)               # sm  -> set(tids)

    # votes
    groups = _read_matches(args.matches)
    ranked = _rank_weight_votes(groups, owners, args.submap, args.per_chip_topk)
    if not ranked:
        raise SystemExit("No matched submaps found (after excluding self).")

    top1_sm, top1_score = ranked[0]
    top2_sm, top2_score = ranked[1] if len(ranked) > 1 else ("", 0.0)

    # tile sets
    q_tids  = sm2tiles.get(args.submap, set())
    m1_tids = sm2tiles.get(top1_sm, set())

    # draw
    out = mosaic.copy()
    query_color = (0, 200, 255)     # BGR (cyan-ish)
    top1_color  = (255, 80, 180)    # BGR (pink/magenta)

    _draw_outline_rects(out, q_tids, nx, cw, ch, query_color, thick=2)
    _draw_outline_rects(out, m1_tids, nx, cw, ch, top1_color,  thick=2)

    # centroids & link
    c_q  = _centroid_from_tiles(q_tids,  nx, cw, ch)
    c_m1 = _centroid_from_tiles(m1_tids, nx, cw, ch)
    if c_q and c_m1:
        p1 = (int(round(c_q[0])),  int(round(c_q[1])))
        p2 = (int(round(c_m1[0])), int(round(c_m1[1])))
        cv2.circle(out, p1, 5, query_color, -1, cv2.LINE_AA)
        cv2.circle(out, p2, 5, top1_color,  -1, cv2.LINE_AA)
        cv2.line(out, p1, p2, (40,40,40), 2, cv2.LINE_AA)
        _draw_label(out, f"{args.submap}", (p1[0]+10, p1[1]-10), color=(255,255,255))
        _draw_label(out, f"{top1_sm}  score={top1_score:.2f}", (p2[0]+10, p2[1]-10), color=(255,255,255))

    # legend panel (stacked below if space; else right)
    legend = _draw_legend()
    # place legend at bottom-right
    lh, lw = legend.shape[:2]
    pad = 10
    y0 = max(0, H - lh - pad)
    x0 = max(0, W - lw - pad)
    roi = out[y0:y0+lh, x0:x0+lw]
    if roi.shape[:2] == legend.shape[:2]:
        alpha = 0.90
        over = (alpha * legend + (1 - alpha) * roi).astype(np.uint8)
        out[y0:y0+lh, x0:x0+lw] = over

    # footer text
    cv2.putText(out, f"Top-2: {top1_sm}({top1_score:.2f})"
                      + (f", {top2_sm}({top2_score:.2f})" if top2_sm else ""),
                (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 2, cv2.LINE_AA)
    cv2.putText(out, f"Top-2: {top1_sm}({top1_score:.2f})"
                      + (f", {top2_sm}({top2_score:.2f})" if top2_sm else ""),
                (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 1, cv2.LINE_AA)

    # write
    if args.out:
        out_path = args.out
    else:
        out_dir = os.path.join(args.root, "anyloc_io", args.submap)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "top1_overlay.png")

    cv2.imwrite(out_path, out)
    print(f"[ok] wrote {out_path}")
    print(f"[info] top-1={top1_sm} ({top1_score:.3f})  top-2={top2_sm or '—'} ({top2_score:.3f})")


if __name__ == "__main__":
    main()
