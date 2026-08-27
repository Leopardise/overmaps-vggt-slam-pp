#!/usr/bin/env python3
import os, json, argparse, glob
import numpy as np, cv2

def dem_to_u8_gray(dem, lo, hi):
    x = dem.astype(np.float32, copy=False)
    vmin, vmax = np.percentile(x, [lo, hi]) if x.size else (0.0, 1.0)
    if vmax - vmin < 1e-12: return np.zeros_like(x, np.uint8)
    t = np.clip((x - vmin) / (vmax - vmin), 0, 1)
    return (t * 255.0 + 0.5).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser("Export submap DEM chips to u8-gray PNGs (consistent with global)")
    ap.add_argument("--root", required=True, help="outputs/run")
    ap.add_argument("--submap", required=True, help="sm_00001 or 'latest'")
    args = ap.parse_args()

    index = json.load(open(os.path.join(args.root, "index.json"), "r"))
    lo, hi = float(index["clip_lo"]), float(index["clip_hi"])

    sm_root = os.path.join(args.root, "submaps")
    if args.submap == "latest":
        subs = sorted([d for d in os.listdir(sm_root) if d.startswith("sm_")])
        if not subs: raise SystemExit("no submaps found")
        sm = subs[-1]
    else:
        sm = args.submap
    chips_dir = os.path.join(sm_root, sm, "chips")
    os.makedirs(chips_dir, exist_ok=True)

    npys = sorted([p for p in glob.glob(os.path.join(chips_dir, "*.npy"))
                   if not p.endswith(".embed.npy")])
    if not npys: raise SystemExit("no chip .npy files found")

    for p in npys:
        u8p = p.replace(".npy", ".png")
        if os.path.isfile(u8p): continue
        dem = np.load(p).astype(np.float32)
        u8 = dem_to_u8_gray(dem, lo, hi)
        cv2.imwrite(u8p, u8)
        # optional small RGB for sanity:
        # cv2.imwrite(p.replace(".npy",".rgb.png"), np.repeat(u8[...,None],3,axis=2))
    print(f"[ok] wrote {len(npys)} PNG chips in {chips_dir}")

if __name__ == "__main__":
    main()
