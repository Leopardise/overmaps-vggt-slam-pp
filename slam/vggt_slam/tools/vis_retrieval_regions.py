#!/usr/bin/env python3
import os, json, argparse, numpy as np, cv2

def main():
    ap = argparse.ArgumentParser("Visualize retrieval regions (top-2 + pad) over mosaic")
    ap.add_argument("--root", required=True)
    ap.add_argument("--submap", required=True)
    ap.add_argument("--io-dir", default="anyloc_io")
    args = ap.parse_args()

    io = os.path.join(args.root, args.io_dir, args.submap)
    reg = json.load(open(os.path.join(io, "region.json")))
    meta = json.load(open(os.path.join(args.root, "index.json")))
    nx, ny = int(meta["nx"]), int(meta["ny"])
    mq = cv2.imread(os.path.join(args.root, "mosaic_quicklook.png"))
    if mq is None:
        raise SystemExit("mosaic_quicklook.png missing")

    H, W = mq.shape[:2]
    cell_w, cell_h = W//nx, H//ny
    overlay = mq.copy()
    alpha = 0.35

    # paint region tiles
    for tid in reg["region_tile_ids"]:
        y, x = divmod(tid, nx)
        x0, y0 = x*cell_w, y*cell_h
        cv2.rectangle(overlay, (x0,y0), (x0+cell_w-1,y0+cell_h-1), (0,165,255), -1)

    out = cv2.addWeighted(overlay, alpha, mq, 1-alpha, 0.0)
    out_p = os.path.join(io, "retrieval_region_overlay.png")
    cv2.imwrite(out_p, out)
    print(f"[ok] {out_p}")

if __name__ == "__main__":
    main()
