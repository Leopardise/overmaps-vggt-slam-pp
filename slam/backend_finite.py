#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finite backend processor for VGGT-SLAM++.
Processes ALL submaps exactly once and exits.
No infinite loops. No watchdog. Safe for automated runs.
"""

import os, glob, argparse
from pathlib import Path
from backend_watch import (
    _read_index,
    _wait_for_ready,
    _load_points_safe,
    _embed_submap_chips_windowed,
    chip_submap_points,
)
import numpy as np
import shutil
import traceback

def process_one(root, sm_dir, do_embeddings, model_name, max_edge, sigma_tiles):
    index_path = os.path.join(root, "index.json")
    if not os.path.isfile(index_path):
        print("[finite-backend] index.json missing, skip.")
        return

    if not _wait_for_ready(sm_dir):
        print(f"[finite-backend] {sm_dir}: READY not found, skip.")
        return

    pts_file = os.path.join(sm_dir, "points_world.npy")
    P_world = _load_points_safe(pts_file)
    if P_world is None or P_world.size == 0:
        print(f"[finite-backend] {sm_dir}: no points.")
        return

    chips_dir = os.path.join(sm_dir, "chips")
    if os.path.isdir(chips_dir):
        shutil.rmtree(chips_dir, ignore_errors=True)
    Path(chips_dir).mkdir(parents=True, exist_ok=True)

    try:
        info = chip_submap_points(
            submap_id=os.path.basename(sm_dir),
            P_world=P_world,
            out_dir=root,
            index_json=index_path,
            reducer="softmax",
            softmax_tau=0.02,
            kernel_px=1.2,
        )
    except Exception:
        print(f"[finite-backend] chipping error: {sm_dir}")
        traceback.print_exc()
        return

    if do_embeddings:
        print(f"[finite-backend] embedding: {sm_dir}")
        try:
            _embed_submap_chips_windowed(
                chips_dir=chips_dir,
                root=root,
                model_name=model_name,
                max_edge=max_edge,
                sigma_tiles=sigma_tiles,
                device="cuda",
                use_amp=True,
            )
        except Exception:
            print(f"[finite-backend] embedding error: {sm_dir}")
            traceback.print_exc()

def main():
    ap = argparse.ArgumentParser("Finite backend DINO windowed embeddings")
    ap.add_argument("--root", required=True)
    ap.add_argument("--with-embeddings", action="store_true")
    ap.add_argument("--dino-model", default="facebook/dinov2-base")
    ap.add_argument("--max-edge", type=int, default=1024)
    ap.add_argument("--sigma-tiles", type=float, default=2.0)
    args = ap.parse_args()

    sm_root = os.path.join(args.root, "submaps")
    if not os.path.isdir(sm_root):
        print("[finite-backend] no submaps directory.")
        return

    submaps = sorted(
        [os.path.join(sm_root, d) for d in os.listdir(sm_root)
         if os.path.isdir(os.path.join(sm_root, d))]
    )

    print(f"[finite-backend] found {len(submaps)} submaps.")

    for sm_dir in submaps:
        process_one(
            root=args.root,
            sm_dir=sm_dir,
            do_embeddings=args.with_embeddings,
            model_name=args.dino_model,
            max_edge=args.max_edge,
            sigma_tiles=args.sigma_tiles,
        )

    print("[finite-backend] DONE.")

if __name__ == "__main__":
    main()
