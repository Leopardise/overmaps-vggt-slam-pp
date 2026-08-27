#!/usr/bin/env python3
"""
VGGT-SLAM++ retry runner for 3 failed scenes.
Fix: --vggt_ckpt points to cached model to avoid HTTP 504 on re-download.
"""

import os, re, glob, time, signal, sys, subprocess
from repo_paths import SLAM_DIR, PYTHON, VGGT_CKPT, OUTPUTS, images_dir

CODE_DIR        = str(SLAM_DIR)
OUT_BASE        = str(OUTPUTS)
LOOP_CLOSURE_PY = os.path.join(CODE_DIR, "vggt_slam/loop_closure.py")

BACKEND_POLL_SEC  = 15
BACKEND_STALL_SEC = 300
BACKEND_MIN_WAIT  = 60

# 3 failed scenes only
SCENES = [
    ("8e2217c3-bb88-45ad-ba3f-7c9d7b3232f5", "Urban_CityStreet_Afternoon_Clear"),
    ("dd329eef-e796-4915-8ce0-dc8e3e844840", "Urban_CityStreet_Night_Clear"),
    ("6dcff26a-ed5d-46c0-b78c-c8af7124dc03", "Interior_Mall_Morning_Indoor"),
]

def log(msg, logfile=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")

def run_cmd(cmd, logfile, cwd=CODE_DIR):
    log(f"CMD: {' '.join(cmd)}", logfile)
    with open(logfile, "a") as lf:
        ret = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        log(f"  WARNING: exited {ret.returncode}", logfile)
    return ret.returncode

def count_ready_submaps(root):
    return len(glob.glob(os.path.join(root, "submaps", "sm_*", "READY")))

def count_embedded_submaps(root):
    done = 0
    for sm_dir in glob.glob(os.path.join(root, "submaps", "sm_*")):
        if os.path.isfile(os.path.join(sm_dir, "READY")):
            if glob.glob(os.path.join(sm_dir, "chips", "*.embed.npy")):
                done += 1
    return done

def wait_for_backend(proc, root, total, logfile):
    log(f"  backend_watch PID={proc.pid} | target: {total} submaps", logfile)
    time.sleep(BACKEND_MIN_WAIT)
    last_count, last_change = 0, time.time()
    while True:
        if proc.poll() is not None:
            log(f"  backend_watch exited (code={proc.returncode})", logfile)
            return
        embedded = count_embedded_submaps(root)
        if embedded != last_count:
            log(f"  backend_watch: {embedded}/{total} embedded", logfile)
            last_count = embedded
            last_change = time.time()
        if embedded >= total:
            log(f"  backend_watch: ALL done. Terminating.", logfile)
            break
        if time.time() - last_change > BACKEND_STALL_SEC:
            log(f"  backend_watch: stalled at {embedded}/{total}. Terminating.", logfile)
            break
        time.sleep(BACKEND_POLL_SEC)
    try:
        proc.send_signal(signal.SIGINT)
        time.sleep(5)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
    except Exception as e:
        log(f"  terminate error: {e}", logfile)
    log("  backend_watch terminated.", logfile)

def patch_loop_closure(root):
    anyloc_io  = os.path.join(root, "anyloc_io")
    votes_path = os.path.join(anyloc_io, "loop_votes.csv")
    pairs_path = os.path.join(anyloc_io, "loop_pairs_ordered.csv")
    with open(LOOP_CLOSURE_PY, "r") as f:
        original = f.read()
    patched = re.sub(r'VOTES_CSV_IN\s*=\s*"[^"]*"',
                     f'VOTES_CSV_IN  = "{votes_path}"', original)
    patched = re.sub(r'PAIRS_CSV_OUT\s*=\s*"[^"]*"',
                     f'PAIRS_CSV_OUT = "{pairs_path}"', patched)
    with open(LOOP_CLOSURE_PY, "w") as f:
        f.write(patched)
    log(f"  [patch] VOTES_CSV_IN  -> {votes_path}")
    log(f"  [patch] PAIRS_CSV_OUT -> {pairs_path}")
    def restore():
        with open(LOOP_CLOSURE_PY, "w") as f:
            f.write(original)
        log("  [restore] loop_closure.py restored")
    return restore

def main_py_args(seq, root, extra=[]):
    """Base args shared by step 1 and step 6. --vggt_ckpt prevents re-download."""
    args = [
        PYTHON, "main.py",
        "--image_folder", seq,
        "--submap_size", "32",
        "--overlapping_window_size", "1",
        "--min_disparity", "40",
        "--use_sim3",
        "--global_dem_out_dir", root,
        "--global_target_px", "90000",
        "--global_tile_px", "4096",
        "--global_reducer", "softmax",
        "--global_softmax_tau", "0.02",
        "--global_cycle_m", "0.001",
        "--global_edge_strength", "0.95",
        "--global_shade_strength", "0.70",
        "--global_dark_level", "0.09",
        "--global_unsharp_radius", "1.0",
        "--global_unsharp_amount", "0.8",
        "--global_clahe_clip", "2.0",
        "--global_clahe_grid", "8",
        "--log_results",
        "--skip_dense_log",
    ] + extra
    if VGGT_CKPT:
        args[3:3] = ["--vggt_ckpt", VGGT_CKPT]
    return args

def run_sequence(uuid, name):
    seq       = str(images_dir(uuid, name))
    root      = os.path.join(OUT_BASE, name)
    anyloc_io = os.path.join(root, "anyloc_io")
    logfile   = os.path.join(OUT_BASE, f"{name}_retry.log")

    # Clean up partial state from previous failed run
    import shutil
    for d in ["submaps", "global_embeddings", "submap_index", "tiles"]:
        p = os.path.join(root, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
            log(f"  Cleaned {p}", logfile)
    for f in glob.glob(os.path.join(root, "*.json")) + \
             glob.glob(os.path.join(root, "*.png")) + \
             glob.glob(os.path.join(root, "*.txt")):
        os.remove(f)

    os.makedirs(root, exist_ok=True)
    os.makedirs(anyloc_io, exist_ok=True)

    log("="*60, logfile)
    log(f"SEQUENCE : {name}", logfile)
    log(f"UUID     : {uuid}", logfile)
    log(f"Root     : {root}", logfile)
    log(f"VGGT_CKPT: {VGGT_CKPT}", logfile)
    log("="*60, logfile)

    # STEP 1
    log("STEP 1: main.py — VGGT + DEM + submap dump", logfile)
    run_cmd(main_py_args(seq, root, [
        "--dump_submaps",
        "--make_global_dem_tiled",
        "--log_path", os.path.join(root, "poses_vanilla.txt"),
    ]), logfile)

    total_submaps = count_ready_submaps(root)
    log(f"STEP 1 done. Submaps: {total_submaps}", logfile)
    if total_submaps == 0:
        log("ERROR: No submaps. Skipping.", logfile)
        return False
    if not os.path.isfile(os.path.join(root, "index.json")):
        log("ERROR: index.json missing. Skipping.", logfile)
        return False

    # STEP 2
    log("STEP 2: backend_watch — chip + DINOv2 embed", logfile)
    with open(logfile, "a") as lf:
        backend_proc = subprocess.Popen(
            [PYTHON, "backend_watch.py",
             "--root", root,
             "--with-embeddings",
             "--dino-model", "facebook/dinov2-base",
             "--max-edge", "1024",
             "--sigma-tiles", "2.0"],
            cwd=CODE_DIR, stdout=lf, stderr=subprocess.STDOUT
        )
    wait_for_backend(backend_proc, root, total_submaps, logfile)
    embedded = count_embedded_submaps(root)
    log(f"STEP 2 done. {embedded}/{total_submaps} embedded.", logfile)
    if embedded == 0:
        log("ERROR: No embeddings. Skipping.", logfile)
        return False

    # STEP 3
    log("STEP 3: embed_global_tiles.py", logfile)
    run_cmd([PYTHON, "vggt_slam/tools/embed_global_tiles.py",
             "--root", root, "--dino-model", "facebook/dinov2-base",
             "--mode", "resize", "--max-edge", "1536", "--overwrite"], logfile)

    # STEP 4
    log("STEP 4: faiss_global_index.py", logfile)
    run_cmd([PYTHON, "vggt_slam/tools/faiss_global_index.py",
             "--root", root, "--index", "hnsw",
             "--hnsw-M", "32", "--hnsw-efC", "200"], logfile)

    # STEP 5
    log("STEP 5a: build_tile_owners.py", logfile)
    run_cmd([PYTHON, "vggt_slam/tools/build_tile_owners.py",
             "--root", root, "--overwrite"], logfile)

    log("STEP 5b: vis_submap_ownership.py", logfile)
    run_cmd([PYTHON, "vggt_slam/tools/vis_submap_ownership.py",
             "--root", root, "--alpha", "0.35", "--grid"], logfile)

    log("STEP 5c: run_batch_submaps.sh", logfile)
    run_cmd(["bash", "vggt_slam/tools/run_batch_submaps.sh",
             root, "0", "20",
             "--dino", "facebook/dinov2-base",
             "--faiss", "hnsw",
             "--match-topk", "20", "--vote-topn", "10", "--vote-pad", "0",
             "--vpr-clusters", "64", "--vpr-topk", "50", "--max-edge", "1024",
             "--force", "--update-loop-votes",
             "--per-chip-topk", "7", "--topn-loop", "7"], logfile)

    votes_csv = os.path.join(anyloc_io, "loop_votes.csv")
    if os.path.isfile(votes_csv):
        with open(votes_csv) as f:
            n = sum(1 for _ in f) - 1
        log(f"  loop_votes.csv: {n} candidate pairs", logfile)
    else:
        log("  WARNING: loop_votes.csv not found", logfile)

    # STEP 6
    log("STEP 6: patch loop_closure.py + optimised trajectory", logfile)
    restore_fn = patch_loop_closure(root)
    poses_out = os.path.join(root, "poses.txt")
    try:
        run_cmd(main_py_args(seq, root, [
            "--max_loops", "7",
            "--log_path", poses_out,
        ]), logfile)
    finally:
        restore_fn()

    if os.path.isfile(poses_out):
        with open(poses_out) as f:
            n = sum(1 for _ in f)
        log(f"STEP 6 done. poses.txt: {n} lines -> {poses_out}", logfile)
    else:
        log("WARNING: poses.txt not found", logfile)

    log(f"SEQUENCE COMPLETE: {name}", logfile)
    log("="*60, logfile)
    return True

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_BASE, exist_ok=True)

    if args.scene:
        scenes_to_run = [(u, n) for u, n in SCENES if n == args.scene]
        if not scenes_to_run:
            print(f"Scene '{args.scene}' not found. Available:")
            for _, n in SCENES: print(f"  {n}")
            sys.exit(1)
    else:
        scenes_to_run = SCENES

    print(f"\nRetrying {len(scenes_to_run)} scene(s):")
    for _, name in scenes_to_run:
        print(f"  {name}")
    print()

    for uuid, name in scenes_to_run:
        try:
            ok = run_sequence(uuid, name)
            print(f"{'✓' if ok else '✗'} {name}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR {name}: {e}")
            import traceback; traceback.print_exc()
            continue

    print("\nAll done.")