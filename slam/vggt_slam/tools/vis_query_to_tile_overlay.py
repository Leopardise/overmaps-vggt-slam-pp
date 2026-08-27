from __future__ import annotations
import os, re, csv, argparse
import numpy as np
import cv2

def read_list(p):
    with open(p, "r") as f:
        return [l.strip() for l in f if l.strip()]

def parse_matches(csv_path, q_paths, d_paths):
    """Return dict qi -> (db_path, score). Robust to either CSV schema."""
    # map path→index (for fallback)
    q_to_i = {p:i for i,p in enumerate(q_paths)}
    d_to_i = {p:i for i,p in enumerate(d_paths)}

    # detect columns
    with open(csv_path, "r") as f:
        header = next(csv.reader(f))
        cols = [c.strip().lower() for c in header]

    have_qidx = ("query_index" in cols) or ("q_index" in cols)
    have_didx = ("db_index" in cols)
    have_paths = ("query_path" in cols) and ("db_path" in cols)

    by_q = {}  # qi -> list[(db_path, score)]
    with open(csv_path, "r") as f:
        rd = csv.DictReader(f)
        for row in rd:
            # score
            try:
                sc = float(row.get("score", "0"))
            except:
                sc = 0.0

            # resolve qi
            if have_qidx:
                key_q = "query_index" if "query_index" in row else "q_index"
                try: qi = int(row[key_q])
                except: continue
                if not (0 <= qi < len(q_paths)): continue
                qp = q_paths[qi]
            else:
                qp = row.get("query_path", "")
                if not qp: continue
                qi = q_to_i.get(qp, None)
                if qi is None:
                    # try basename fallback
                    base = os.path.basename(qp)
                    qi = next((i for i,p in enumerate(q_paths) if os.path.basename(p)==base), None)
                    if qi is None: continue

            # resolve db path
            if have_didx and row.get("db_index","")!="":
                try: di = int(row["db_index"])
                except: continue
                if not (0 <= di < len(d_paths)): continue
                dp = d_paths[di]
            else:
                dp = row.get("db_path", "")
                if not dp:
                    # attempt basename match
                    continue
            by_q.setdefault(qi, []).append((dp, sc))

    # take best per query
    best = {}
    for qi, lst in by_q.items():
        if not lst: continue
        lst.sort(key=lambda x: -x[1])
        best[qi] = lst[0]  # (db_path, score)
    return best

def imread_rgb_safe(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None: return None
    if im.ndim == 2:
        im = np.repeat(im[...,None], 3, axis=2)
    elif im.shape[2]==4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def distinct_colors(n):
    """HSV wheel → BGR uint8."""
    if n <= 0: return []
    hsv = np.zeros((n,1,3), np.uint8)
    for i in range(n):
        hsv[i,0,0] = int(180.0*i/max(1,n))       # H
        hsv[i,0,1] = 200                         # S
        hsv[i,0,2] = 230                         # V
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n,3)
    return [tuple(int(c) for c in bgr[i]) for i in range(n)]

def tile_id_from_path(p):
    m = re.search(r"tile_(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser("Overlay best query→tile matches")
    ap.add_argument("--root", required=True)            # outputs/run
    ap.add_argument("--io-dir", required=True)          # outputs/run/anyloc_io/sm_xxx
    ap.add_argument("--matches", required=True)         # matches_*.csv
    ap.add_argument("--max-queries", type=int, default=9999)
    ap.add_argument("--query-cell", type=int, default=160)
    ap.add_argument("--tile-alpha", type=float, default=0.55)
    ap.add_argument("--outline", type=int, default=3)
    args = ap.parse_args()

    # IO files
    qtxt = os.path.join(args.io_dir, "queries.txt")
    dtxt = os.path.join(args.io_dir, "database.txt")
    mq_png = os.path.join(args.root, "mosaic_quicklook.png")
    idx_json = os.path.join(args.root, "index.json")

    if not (os.path.isfile(qtxt) and os.path.isfile(dtxt) and os.path.isfile(mq_png) and os.path.isfile(idx_json)):
        print("[err] missing one of queries.txt / database.txt / mosaic_quicklook.png / index.json")
        return

    import json
    q_paths = read_list(qtxt)
    d_paths = read_list(dtxt)
    meta = json.load(open(idx_json,"r"))
    nx, ny   = int(meta["nx"]), int(meta["ny"])

    best = parse_matches(args.matches, q_paths, d_paths)
    if not best:
        print("[err] could not parse any matches from CSV.")
        return

    # load mosaic and compute per-tile pixel size
    mosaic = cv2.imread(mq_png, cv2.IMREAD_COLOR)
    Hm, Wm = mosaic.shape[:2]
    tw = Wm // nx
    th = Hm // ny

    # choose colors for queries (in query index order)
    q_used = sorted(best.keys())[:args.max_queries]
    cols = distinct_colors(len(q_used))

    # left strip with query chips
    strip_w = args.query_cell + 16
    canvas = np.full((Hm, strip_w + Wm, 3), 245, np.uint8)
    canvas[:, strip_w:strip_w+Wm] = mosaic

    # draw each query chip in strip + color matched tile on mosaic
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, qi in enumerate(q_used):
        dp, sc = best[qi]
        color = cols[i]
        # draw query chip
        qim = imread_rgb_safe(q_paths[qi])
        if qim is None:
            qim = np.full((args.query_cell, args.query_cell,3), 220, np.uint8)
        else:
            qim = cv2.resize(qim[:,:,::-1], (args.query_cell,args.query_cell), interpolation=cv2.INTER_AREA)  # RGB->BGR
        y0 = int(i * (Hm / max(1,len(q_used))))
        y1 = min(Hm, y0 + args.query_cell)
        # ensure room
        if y1 - y0 < args.query_cell:
            y0 = max(0, y1 - args.query_cell)
        canvas[y0:y0+args.query_cell, 8:8+args.query_cell] = qim
        cv2.rectangle(canvas, (8, y0+0), (8+args.query_cell-1, y0+args.query_cell-1), color, 2)
        lbl = f"Q{qi}  s={sc:.3f}"
        cv2.putText(canvas, lbl, (8, y0+args.query_cell+16), font, 0.5, (30,30,30), 1, cv2.LINE_AA)

        # color the matched global tile
        tid = tile_id_from_path(dp)
        if tid is None: continue
        ty, tx = divmod(tid, nx)
        x0 = strip_w + tx*tw
        y0t = ty*th
        # translucent fill
        roi = canvas[y0t:y0t+th, x0:x0+tw]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0,0), (tw-1, th-1), color, -1)
        cv2.addWeighted(overlay, args.tile_alpha, roi, 1-args.tile_alpha, 0, dst=roi)
        # outline
        cv2.rectangle(canvas, (x0, y0t), (x0+tw-1, y0t+th-1), color, args.outline)
        # tiny label
        cv2.putText(canvas, f"Q{qi}", (x0+6, y0t+18), font, 0.5, (20,20,20), 1, cv2.LINE_AA)

    out_png = os.path.join(args.io_dir, "query_to_tile_overlay.png")
    cv2.imwrite(out_png, canvas)
    print(f"[ok] wrote {out_png}")
    print("[note] Left strip: query chips framed by color. Global DEM: matched tiles filled/outlined with same color.")
    print("[tip] If a tile looks too small, zoom in; this uses the prebuilt mosaic_quicklook.png size.")
    
if __name__ == "__main__":
    main()
