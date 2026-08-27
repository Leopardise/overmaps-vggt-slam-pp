from __future__ import annotations
import os, json, argparse
import numpy as np
import cv2
from collections import defaultdict

# ---------- helpers ----------

def _read_json(p): 
    with open(p, "r") as f: 
        return json.load(f)

def _hsv_to_bgr(h, s, v):
    """h in [0,1), s,v in [0,1] → BGR uint8"""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if   i == 0: r,g,b = v,t,p
    elif i == 1: r,g,b = q,v,p
    elif i == 2: r,g,b = p,v,t
    elif i == 3: r,g,b = p,q,v
    elif i == 4: r,g,b = t,p,v
    else:        r,g,b = v,p,q
    return np.array([b,g,r]) * 255.0

def _distinct_palette(n, sat=0.75, val=0.95, seed_offset=0.07):
    """Evenly spaced hues → distinct bright colors (BGR uint8)."""
    cols = []
    for k in range(n):
        h = (seed_offset + k / max(1, n)) % 1.0
        cols.append(_hsv_to_bgr(h, sat, val).astype(np.uint8))
    return cols

# ---------- core ----------

def main():
    ap = argparse.ArgumentParser("Visualize per-submap scores on global mosaic with legend")
    ap.add_argument("--root", required=True)
    ap.add_argument("--reg-submap", required=True, help="registering submap id, e.g. sm_00001")
    ap.add_argument("--votes", default="", help="override path to faiss_votes_by_submap.json")
    ap.add_argument("--owners", default="", help="override path to submap_index/tile_owners.json")
    ap.add_argument("--combine", default="max", choices=["max","sum"],
                    help="if a tile has multiple owners, combine their scores by max or sum")
    ap.add_argument("--topN-legend", type=int, default=8, help="show top-N winning submaps in legend")
    ap.add_argument("--fill-alpha-min", type=float, default=0.25, help="min opacity for a scored tile")
    ap.add_argument("--fill-alpha-max", type=float, default=0.75, help="max opacity for highest-scored tile")
    ap.add_argument("--outline-thick", type=int, default=1, help="outline thickness (0 = off)")
    ap.add_argument("--grid", action="store_true", help="draw faint tile grid for reference")
    args = ap.parse_args()

    root = args.root
    reg  = args.reg_submap
    votes_p  = args.votes  or os.path.join(root, "submaps", reg, "faiss_votes_by_submap.json")
    owners_p = args.owners or os.path.join(root, "submap_index", "tile_owners.json")
    meta_p   = os.path.join(root, "index.json")
    mosaic_p = os.path.join(root, "mosaic_quicklook.png")

    if not os.path.isfile(votes_p):  raise SystemExit(f"[err] missing: {votes_p}")
    if not os.path.isfile(owners_p): raise SystemExit(f"[err] missing: {owners_p}")
    if not os.path.isfile(meta_p):   raise SystemExit(f"[err] missing: {meta_p}")
    if not os.path.isfile(mosaic_p): raise SystemExit(f"[err] missing: {mosaic_p}")

    meta = _read_json(meta_p)
    nx, ny = int(meta["nx"]), int(meta["ny"])
    mosaic = cv2.imread(mosaic_p, cv2.IMREAD_COLOR)
    Hm, Wm = mosaic.shape[:2]
    tile_w = Wm / nx
    tile_h = Hm / ny

    votes = _read_json(votes_p)
    reg_in = votes.get("registering_submap", reg)
    if reg_in != reg:
        print(f"[warn] votes are for {reg_in}, you passed {reg}. Proceeding anyway.")

    # submap → score
    ranking = votes.get("ranking", [])
    submap_score = {sm: float(sc) for sm, sc in ranking}
    owners = _read_json(owners_p).get("tile_owners", {})

    # per-tile score + winning submap
    tile_scores = np.zeros(nx*ny, np.float32)
    tile_winner = np.full(nx*ny, "", dtype=object)
    for tid_str, subs in owners.items():
        tid = int(tid_str)
        if tid < 0 or tid >= nx*ny or not subs:
            continue
        vals = [submap_score.get(sm, 0.0) for sm in subs]
        if args.combine == "sum":
            val = float(np.sum(vals))
            win_idx = int(np.argmax(vals)) if len(vals) else -1
        else:  # max
            win_idx = int(np.argmax(vals)) if len(vals) else -1
            val = float(vals[win_idx]) if win_idx >= 0 else 0.0
        tile_scores[tid] = val
        tile_winner[tid] = subs[win_idx] if win_idx >= 0 else ""

    # normalize scores to [0,1] for opacity scaling
    if tile_scores.max() > 0:
        ts = tile_scores / (tile_scores.max() + 1e-9)
    else:
        ts = tile_scores.copy()

    # pick top-N winners for categorical colors (others → light gray)
    # score by *total contribution* across tiles they win
    win_contrib = defaultdict(float)
    for tid in range(nx*ny):
        sm = tile_winner[tid]
        if sm:
            win_contrib[sm] += tile_scores[tid]
    top = sorted(win_contrib.items(), key=lambda kv: -kv[1])[:max(1, args.topN_legend)]
    top_submaps = [sm for sm, _ in top]
    palette = _distinct_palette(len(top_submaps), sat=0.80, val=0.95)
    color_of = {sm: palette[i].tolist() for i, sm in enumerate(top_submaps)}

    # build overlay
    overlay = mosaic.copy()

    # semi-transparent fill per tile (alpha scales with normalized score)
    for tid in range(nx*ny):
        sc = ts[tid]
        if sc <= 0: 
            continue
        sm = tile_winner[tid]
        if not sm:
            continue
        color = color_of.get(sm, [200, 200, 200])  # gray for non-top submaps
        alpha = args.fill_alpha_min + (args.fill_alpha_max - args.fill_alpha_min) * float(sc)
        ty, tx = divmod(tid, nx)
        x0 = int(round(tx * tile_w)); y0 = int(round(ty * tile_h))
        x1 = int(round((tx + 1) * tile_w)); y1 = int(round((ty + 1) * tile_h))
        roi = overlay[y0:y1, x0:x1]
        fill = np.zeros_like(roi) + np.asarray(color, np.uint8)
        cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0, dst=roi)

    # optional outlines
    if args.outline_thick > 0:
        for tid in range(nx*ny):
            sc = ts[tid]
            if sc <= 0: 
                continue
            sm = tile_winner[tid]
            color = color_of.get(sm, [180, 180, 180])
            ty, tx = divmod(tid, nx)
            x0 = int(round(tx * tile_w)); y0 = int(round(ty * tile_h))
            x1 = int(round((tx + 1) * tile_w)); y1 = int(round((ty + 1) * tile_h))
            cv2.rectangle(overlay, (x0, y0), (x1-1, y1-1), color, thickness=args.outline_thick, lineType=cv2.LINE_AA)

    # optional faint grid
    if args.grid:
        grid_col = (230, 230, 230)
        for tx in range(1, nx):
            x = int(round(tx * tile_w))
            cv2.line(overlay, (x, 0), (x, Hm-1), grid_col, 1)
        for ty in range(1, ny):
            y = int(round(ty * tile_h))
            cv2.line(overlay, (0, y), (Wm-1, y), grid_col, 1)

    # ---- legend panel on the right ----
    pad = 12
    sw  = 22   # swatch size
    lh  = 26   # line height
    legend_w = 340
    legend_h = pad*2 + lh * len(top_submaps)
    H_out = max(Hm, legend_h + 2*pad)
    W_out = Wm + legend_w
    canvas = np.full((H_out, W_out, 3), 255, np.uint8)
    canvas[:Hm, :Wm] = overlay

    xL = Wm + pad
    yL = pad + 8
    title = f"Matches vs {reg}"
    cv2.putText(canvas, title, (xL, yL), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(canvas, title, (xL, yL), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    y = yL + 16 + pad

    # compute normalized legend scores (relative to max submap score)
    if top:
        max_s = max([win_contrib[sm] for sm in top_submaps]) + 1e-9
    else:
        max_s = 1.0

    for sm in top_submaps:
        col = color_of[sm]
        sc_abs = win_contrib[sm]
        sc_rel = sc_abs / max_s
        # swatch
        cv2.rectangle(canvas, (xL, y), (xL+sw, y+sw), col, -1)
        # border
        cv2.rectangle(canvas, (xL, y), (xL+sw, y+sw), (0,0,0), 1)
        # label
        label = f"{sm}   score={sc_abs:.1f}   ({sc_rel*100:.0f}%)"
        cv2.putText(canvas, label, (xL + sw + 10, y + sw - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (xL + sw + 10, y + sw - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        y += lh

    out_png = os.path.join(root, "submaps", reg, "submap_scores_overlay.png")
    cv2.imwrite(out_png, canvas)
    print(f"[ok] wrote {out_png}")
    print("[hint] color = winning submap; opacity scales with per-tile score. Legend lists top-N winners.")

if __name__ == "__main__":
    main()
