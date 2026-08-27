#!/usr/bin/env python3
"""VGGT-SLAM++ back-end (paper Fig. 3, Sec. 3).

Pipeline, in order:
  1. Embed global DEM tiles with DINOv2 + 9×9 Gaussian/visibility weights (Eq. 1)
  2. Build a FAISS-HNSW index (Eq. 2)
  3. For each query submap: chip at 2 m, embed, vote to parent submaps (Eq. 3)
  4. Restrict AnyLoc VPR to the covisibility window
  5. Write loop_votes.csv for Sim(3) back-end optimisation (Eq. 4)

Off-the-shelf DINOv2 weights are downloaded by torch.hub on first use.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser("VGGT-SLAM++ DEM / VPR / Sim(3) back-end")
    ap.add_argument("--root", required=True, help="run directory with index.json, tiles/, submaps/")
    ap.add_argument("--dino-model", default="facebook/dinov2-base")
    ap.add_argument("--window-radius", type=int, default=4, help="4 → 9×9 neighbourhood (Eq. 1)")
    ap.add_argument("--sigma-tiles", type=float, default=2.0)
    ap.add_argument("--max-edge", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=10, help="top-K covisible submaps (paper: 10)")
    ap.add_argument("--with-vpr", action="store_true", help="run AnyLoc-style VLAD VPR inside the covis window")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    py = sys.executable

    run([
        py, "-m", "vggt_slam.tools.embed_global_tiles",
        "--root", root,
        "--dino-model", args.dino_model,
        "--window-radius", str(args.window_radius),
        "--sigma-tiles", str(args.sigma_tiles),
        "--max-edge", str(args.max_edge),
    ])
    run([py, "-m", "vggt_slam.tools.faiss_global_index", "--root", root])
    run([py, "-m", "vggt_slam.tools.build_tile_owners", "--root", root])

    submaps = sorted(glob.glob(os.path.join(root, "submaps", "sm_*")))
    for sm_dir in submaps:
        sm = os.path.basename(sm_dir)
        run([
            py, "backend_finite.py",
            "--root", root,
            "--with-embeddings",
            "--dino-model", args.dino_model,
            "--max-edge", str(args.max_edge),
            "--sigma-tiles", str(args.sigma_tiles),
        ])
        break  # backend_finite already iterates all submaps once

    for sm_dir in submaps:
        sm = os.path.basename(sm_dir)
        matches = os.path.join(sm_dir, "matches_topk.json")
        if os.path.isfile(matches):
            run([
                py, "-m", "vggt_slam.tools.submap_faiss_vote",
                "--root", root, "--submap", sm, "--weighted", "--exclude-self",
            ])
            run([
                py, "-m", "vggt_slam.tools.prepare_anyloc_io",
                "--root", root, "--submap", sm,
            ])
            if args.with_vpr:
                io_dir = os.path.join(root, "anyloc_io", sm)
                q = os.path.join(io_dir, "queries.txt")
                d = os.path.join(io_dir, "database.txt")
                if os.path.isfile(q) and os.path.isfile(d):
                    run([
                        py, "-m", "vggt_slam.tools.run_vpr_retrieval",
                        "--queries", q, "--database", d,
                        "--topk", str(args.topk),
                        "--backbone", args.dino_model,
                        "--out", os.path.join(io_dir, "matches_anyloc.csv"),
                    ])
                    run([
                        py, "-m", "vggt_slam.tools.update_loop_votes_csv",
                        "--root", root, "--submap", sm,
                    ])

    votes = os.path.join(root, "anyloc_io", "loop_votes.csv")
    if os.path.isfile(votes):
        print(f"[ok] loop votes → {votes}")
        print("Re-run the front-end with --external_loops_csv", votes, "or apply with vggt_slam.anyloc_csv")
    else:
        print("[warn] loop_votes.csv not written yet; check FAISS votes / VPR outputs")


if __name__ == "__main__":
    main()
