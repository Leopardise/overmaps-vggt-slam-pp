from __future__ import annotations
import os, csv, argparse, math
import cv2
import numpy as np

def read_list(txt_path):
    with open(txt_path, "r") as f:
        return [l.strip() for l in f if l.strip()]

def load_matches(csv_path, q_paths, d_paths, topk, max_queries):
    """
    Return a list of length nq_show:
      rows[i] = list of (db_path, score) of length <= topk
    """
    have_qidx = False
    have_paths = False

    # Peek header
    with open(csv_path, "r") as f:
        r = csv.reader(f)
        header = next(r)
        cols = [h.strip() for h in header]
        have_qidx = ("query_index" in cols) or ("q_index" in cols)
        have_paths = ("query_path" in cols) and ("db_path" in cols)

    # Map path->index for fallback
    q_index = {p: i for i, p in enumerate(q_paths)}
    d_index = {p: i for i, p in enumerate(d_paths)}

    # Collect rows grouped by query index
    per_q = {}  # qi -> list[(db_path, score)]
    with open(csv_path, "r") as f:
        rd = csv.DictReader(f)
        for row in rd:
            # figure out qi, dp, score
            if have_qidx:
                key_q = "query_index" if "query_index" in row else "q_index"
                qi = int(row[key_q])
                if 0 <= qi < len(q_paths):
                    qp = q_paths[qi]
                else:
                    continue
                if "db_index" in row and row["db_index"] != "":
                    di = int(row["db_index"])
                    if 0 <= di < len(d_paths):
                        dp = d_paths[di]
                    else:
                        continue
                else:
                    # need db_path
                    if have_paths and row.get("db_path", ""):
                        dp = row["db_path"]
                    else:
                        continue
            else:
                # need paths in CSV; if absent, skip
                qp = row.get("query_path", "")
                dp = row.get("db_path", "")
                if not qp or not dp:
                    # allow a fallback where only filenames are present:
                    # try to match by basename
                    continue
                # map path to index to keep grouping consistent with queries.txt order
                qi = q_index.get(qp, None)
                if qi is None:
                    # try basename match
                    base = os.path.basename(qp)
                    qi = next((i for i,p in enumerate(q_paths) if os.path.basename(p)==base), None)
                    if qi is None:
                        continue

            try:
                sc = float(row.get("score", "0"))
            except:
                sc = 0.0

            per_q.setdefault(qi, []).append((dp, sc))

    # Keep only topk per query by score, and at most max_queries queries (in query order)
    q_sorted = sorted([qi for qi in per_q.keys() if 0 <= qi < len(q_paths)])
    if max_queries > 0:
        q_sorted = q_sorted[:max_queries]

    out = []
    q_used = []
    for qi in q_sorted:
        lst = per_q[qi]
        lst.sort(key=lambda x: -x[1])
        out.append(lst[:topk])
        q_used.append(qi)

    return out, q_used

def imread_rgb_safe(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, axis=2)
    elif im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def tile_text(img_rgb, text, bg_alpha=0.65):
    h, w = img_rgb.shape[:2]
    overlay = img_rgb.copy()
    cv2.rectangle(overlay, (0,0), (w, 24), (0,0,0), -1)
    out = cv2.addWeighted(overlay, bg_alpha, img_rgb, 1-bg_alpha, 0)
    cv2.putText(out, text, (6,17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def make_grid(q_used, rows, q_paths, Wcell=224, Hcell=224, topk=5, pad=4):
    """
    rows: list over queries, each element is list of (db_path, score)
    Render grid with (topk+1) columns: first = query, others = matches.
    """
    nQ = len(rows)
    if nQ == 0:
        return None

    H = nQ * (Hcell + pad) + pad
    W = (topk + 1) * (Wcell + pad) + pad
    canvas = np.full((H, W, 3), 245, np.uint8)

    for r, (qi, lst) in enumerate(zip(q_used, rows)):
        # Query cell
        qp = q_paths[qi]
        qim = imread_rgb_safe(qp)
        if qim is None:
            qim = np.full((Hcell, Wcell, 3), 230, np.uint8)
            txt = f"Q{qi}: [missing]"
        else:
            qim = cv2.resize(qim, (Wcell, Hcell), interpolation=cv2.INTER_AREA)
            txt = f"Q{qi}: {os.path.basename(qp)}"
        qim = tile_text(qim, txt)
        y0 = pad + r*(Hcell+pad)
        x0 = pad
        canvas[y0:y0+Hcell, x0:x0+Wcell] = qim[:, :, ::-1]  # back to BGR for saving

        # Matches
        for c in range(topk):
            x0 = pad + (c+1)*(Wcell+pad)
            if c < len(lst):
                dp, sc = lst[c]
                dim = imread_rgb_safe(dp)
                if dim is None:
                    dim = np.full((Hcell, Wcell, 3), 200, np.uint8)
                    txt = f"{os.path.basename(dp)}  s={sc:.3f}  [missing]"
                else:
                    dim = cv2.resize(dim, (Wcell, Hcell), interpolation=cv2.INTER_AREA)
                    txt = f"{os.path.basename(dp)}  s={sc:.3f}"
                dim = tile_text(dim, txt)
                canvas[y0:y0+Hcell, x0:x0+Wcell] = dim[:, :, ::-1]
            else:
                # empty cell
                canvas[y0:y0+Hcell, x0:x0+Wcell] = np.full((Hcell, Wcell, 3), 255, np.uint8)
    return canvas

def main():
    ap = argparse.ArgumentParser("Visualize per-chip matches")
    ap.add_argument("--root", required=True, help="e.g. outputs/run (unused, for consistency/logging)")
    ap.add_argument("--io-dir", required=True, help="e.g. outputs/run/anyloc_io/sm_00001")
    ap.add_argument("--matches", required=True, help="CSV from retrieval")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=24)
    ap.add_argument("--cell", type=int, default=224)
    args = ap.parse_args()

    qtxt = os.path.join(args.io_dir, "queries.txt")
    dtxt = os.path.join(args.io_dir, "database.txt")
    if not (os.path.isfile(qtxt) and os.path.isfile(dtxt)):
        print(f"[err] missing queries.txt/database.txt under {args.io_dir}")
        return

    q_paths = read_list(qtxt)
    d_paths = read_list(dtxt)

    rows, q_used = load_matches(args.matches, q_paths, d_paths, args.topk, args.max_queries)
    if not rows:
        print("[err] nothing to visualize (no matches parsed).")
        return

    grid = make_grid(q_used, rows, q_paths, Wcell=args.cell, Hcell=args.cell, topk=args.topk)
    out_png = os.path.join(args.io_dir, "patch_matches_grid.png")
    cv2.imwrite(out_png, grid)
    print(f"[ok] wrote {out_png}")

if __name__ == "__main__":
    main()
