#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, glob
from pathlib import Path
from subprocess import run

PKG = "vggt_slam.backend"  # package-qualified

def runm(mod: str, *args: str):
    cmd = ["python", "-m", f"{PKG}.{mod}", *args]
    run(cmd, check=True)

def main(out_root: str, poll_sec: float = 2.0):
    emb_built = os.path.exists(os.path.join(out_root, "global_embeddings", "faiss_hnsw.index"))

    while True:
        tiles_ready = os.path.exists(os.path.join(out_root, "index.json")) and \
                      glob.glob(os.path.join(out_root, "tiles", "tile_*.npy"))
        if tiles_ready and not emb_built:
            print("[watch] building global embeddings + FAISS…")
            runm("embed_global_tiles", "--out_root", out_root)
            runm("build_faiss", "--emb_dir", os.path.join(out_root, "global_embeddings"))
            emb_built = True

        for sm_ready in sorted(glob.glob(os.path.join(out_root, "submaps", "sm_*", "READY"))):
            sm_dir = str(Path(sm_ready).parent)
            done_flag = os.path.join(sm_dir, "MATCHED")
            if os.path.exists(done_flag):
                continue

            print(f"[watch] processing {sm_dir}")
            runm("submap_chipper", "--out_root", out_root, "--sm_dir", sm_dir)
            runm("make_embeddings", "--chips_dir", os.path.join(sm_dir, "chips"))
            runm("submap_chip_embed_match", "--out_root", out_root, "--sm_dir", sm_dir, "--topk", "20")
            Path(done_flag).write_text("ok\n")

        time.sleep(poll_sec)

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--poll_sec", type=float, default=2.0)
    args = ap.parse_args()
    main(args.out_root, args.poll_sec)
