#!/usr/bin/env python3
"""
Insta360 360-degree data pipeline for VGGT-SLAM++.

Step 0: Convert equirectangular frames -> perspective crops
Step 1-6: Full VGGT-SLAM++ pipeline (same as OverMaps)

Equirectangular (3840x2160, 2:1) cannot be fed directly to VGGT.
We extract a forward-facing perspective crop from each frame.
FoV: 90 degrees horizontal, output: 1024x1024 (square, compatible with VGGT)
"""

import os, re, glob, time, signal, sys, subprocess
import numpy as np
import cv2

# ── CONFIG ────────────────────────────────────────────────────────────────────

from repo_paths import SLAM_DIR, PYTHON, VGGT_CKPT, INSTA, OUTPUTS

CODE_DIR        = str(SLAM_DIR)
INSTA_BASE      = str(INSTA)
OUT_BASE        = str(OUTPUTS)
LOOP_CLOSURE_PY = os.path.join(CODE_DIR, "vggt_slam/loop_closure.py")

# Perspective crop settings
CROP_FOV_DEG  = 90    # horizontal FoV in degrees
CROP_SIZE     = 1024  # output square size (VGGT-friendly)
CROP_YAW_DEG  = 0     # 0 = forward facing
CROP_PITCH_DEG = 0    # 0 = level horizon

BACKEND_POLL_SEC  = 15
BACKEND_STALL_SEC = 300
BACKEND_MIN_WAIT  = 60

# ── 3 CLIPS ───────────────────────────────────────────────────────────────────

CLIPS = [
    ("239eef93", "Insta360_Clip1_39frames",  "Venice (39-series, 46 frames)"),
    ("2514321c", "Insta360_Clip2_77frames",  "Venice (77-series, 94 frames)"),
    ("342137b5", "Insta360_Clip3_489frames", "Venice (489-series, 83 frames)"),
]

# ── EQUIRECTANGULAR → PERSPECTIVE ────────────────────────────────────────────

def equirect_to_perspective(equirect_img, fov_deg, yaw_deg, pitch_deg, out_size):
    """
    Extract a perspective crop from an equirectangular image.
    fov_deg: horizontal field of view
    yaw_deg: pan left/right (0 = forward)
    pitch_deg: tilt up/down (0 = level)
    out_size: output image size (square)
    """
    h, w = equirect_img.shape[:2]
    f = out_size / (2 * np.tan(np.radians(fov_deg) / 2))

    # Output pixel coordinates
    cx = cy = out_size / 2
    u, v = np.meshgrid(np.arange(out_size), np.arange(out_size))
    x = (u - cx) / f
    y = (v - cy) / f
    z = np.ones_like(x)

    # Normalize
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/norm, y/norm, z/norm

    # Rotate by yaw (around Y axis)
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Pitch rotation (around X axis)
    y2 = y * np.cos(pitch) - z * np.sin(pitch)
    z2 = y * np.sin(pitch) + z * np.cos(pitch)
    y, z = y2, z2

    # Yaw rotation (around Y axis)
    x2 = x * np.cos(yaw) + z * np.sin(yaw)
    z2 = -x * np.sin(yaw) + z * np.cos(yaw)
    x, z = x2, z2

    # Convert to spherical coords
    lon = np.arctan2(x, z)  # [-pi, pi]
    lat = np.arcsin(np.clip(y, -1, 1))  # [-pi/2, pi/2]

    # Map to equirectangular pixel coords
    map_x = ((lon / np.pi + 1) / 2 * w).astype(np.float32)
    map_y = ((-lat / (np.pi/2) + 1) / 2 * h).astype(np.float32)

    return cv2.remap(equirect_img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)

def convert_clip_to_perspective(clip_id, name, logfile=None):
    """Convert all equirectangular frames to perspective crops."""
    src_dir  = os.path.join(INSTA_BASE, clip_id, "images")
    out_root = os.path.join(OUT_BASE, name)
    persp_dir = os.path.join(out_root, "perspective_images")
    os.makedirs(persp_dir, exist_ok=True)

    frames = sorted([f for f in os.listdir(src_dir) if f.endswith('.jpg')])
    log(f"  Converting {len(frames)} equirectangular frames -> {CROP_SIZE}x{CROP_SIZE} perspective crops", logfile)
    log(f"  FoV={CROP_FOV_DEG}° yaw={CROP_YAW_DEG}° pitch={CROP_PITCH_DEG}°", logfile)

    for i, fname in enumerate(frames):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(persp_dir, fname)
        if os.path.exists(dst):
            continue
        img = cv2.imread(src)
        if img is None:
            log(f"  WARNING: could not read {src}", logfile)
            continue
        crop = equirect_to_perspective(img, CROP_FOV_DEG, CROP_YAW_DEG,
                                       CROP_PITCH_DEG, CROP_SIZE)
        cv2.imwrite(dst, crop)

    converted = len([f for f in os.listdir(persp_dir) if f.endswith('.jpg')])
    log(f"  Converted {converted}/{len(frames)} frames -> {persp_dir}", logfile)
    return persp_dir

# ── HELPERS (same as OverMaps runner) ────────────────────────────────────────

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
    def restore():
        with open(LOOP_CLOSURE_PY, "w") as f:
            f.write(original)
        log("  [restore] loop_closure.py restored")
    return restore

def main_args(seq, root, extra=[]):
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

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_clip(clip_id, name, desc):
    root      = os.path.join(OUT_BASE, name)
    anyloc_io = os.path.join(root, "anyloc_io")
    logfile   = os.path.join(OUT_BASE, f"{name}.log")

    os.makedirs(root, exist_ok=True)
    os.makedirs(anyloc_io, exist_ok=True)

    log("="*60, logfile)
    log(f"CLIP     : {name}", logfile)
    log(f"DESC     : {desc}", logfile)
    log(f"Root     : {root}", logfile)
    log("="*60, logfile)

    # STEP 0: equirectangular -> perspective conversion
    log("STEP 0: equirectangular -> perspective crop conversion", logfile)
    src_dir = os.path.join(INSTA_BASE, clip_id, "images")
    if not os.path.isdir(src_dir) or len(os.listdir(src_dir)) == 0:
        log(f"ERROR: No images found at {src_dir}. Skipping.", logfile)
        return False
    seq = convert_clip_to_perspective(clip_id, name, logfile)

    # STEP 1: VGGT + DEM + submap dump
    log("STEP 1: main.py — VGGT + DEM + submap dump", logfile)
    run_cmd(main_args(seq, root, [
        "--dump_submaps", "--make_global_dem_tiled",
        "--log_path", os.path.join(root, "poses_vanilla.txt"),
    ]), logfile)

    total = count_ready_submaps(root)
    log(f"STEP 1 done. Submaps: {total}", logfile)
    if total == 0:
        log("ERROR: No submaps. Skipping.", logfile)
        return False

    # STEP 2: backend_watch
    log("STEP 2: backend_watch — chip + DINOv2 embed", logfile)
    with open(logfile, "a") as lf:
        backend_proc = subprocess.Popen(
            [PYTHON, "backend_watch.py",
             "--root", root, "--with-embeddings",
             "--dino-model", "facebook/dinov2-base",
             "--max-edge", "1024", "--sigma-tiles", "2.0"],
            cwd=CODE_DIR, stdout=lf, stderr=subprocess.STDOUT
        )
    wait_for_backend(backend_proc, root, total, logfile)
    embedded = count_embedded_submaps(root)
    log(f"STEP 2 done. {embedded}/{total} embedded.", logfile)

    # STEP 3: embed global tiles
    log("STEP 3: embed_global_tiles.py", logfile)
    run_cmd([PYTHON, "vggt_slam/tools/embed_global_tiles.py",
             "--root", root, "--dino-model", "facebook/dinov2-base",
             "--mode", "resize", "--max-edge", "1536", "--overwrite"], logfile)

    # STEP 4: FAISS
    log("STEP 4: faiss_global_index.py", logfile)
    run_cmd([PYTHON, "vggt_slam/tools/faiss_global_index.py",
             "--root", root, "--index", "hnsw",
             "--hnsw-M", "32", "--hnsw-efC", "200"], logfile)

    # STEP 5: tile owners + vis + loop candidates
    log("STEP 5a: build_tile_owners.py", logfile)
    run_cmd([PYTHON, "vggt_slam/tools/build_tile_owners.py",
             "--root", root, "--overwrite"], logfile)
    log("STEP 5b: vis_submap_ownership.py", logfile)
    run_cmd([PYTHON, "vggt_slam/tools/vis_submap_ownership.py",
             "--root", root, "--alpha", "0.35", "--grid"], logfile)
    log("STEP 5c: run_batch_submaps.sh", logfile)
    run_cmd(["bash", "vggt_slam/tools/run_batch_submaps.sh",
             root, "0", "20",
             "--dino", "facebook/dinov2-base", "--faiss", "hnsw",
             "--match-topk", "20", "--vote-topn", "10", "--vote-pad", "0",
             "--vpr-clusters", "64", "--vpr-topk", "50", "--max-edge", "1024",
             "--force", "--update-loop-votes",
             "--per-chip-topk", "7", "--topn-loop", "7"], logfile)

    votes_csv = os.path.join(anyloc_io, "loop_votes.csv")
    if os.path.isfile(votes_csv):
        with open(votes_csv) as f:
            n = sum(1 for _ in f) - 1
        log(f"  loop_votes.csv: {n} pairs", logfile)

    # STEP 6: patch + optimised trajectory
    log("STEP 6: patch loop_closure.py + optimised trajectory", logfile)
    restore_fn = patch_loop_closure(root)
    poses_out = os.path.join(root, "poses.txt")
    try:
        run_cmd(main_args(seq, root, [
            "--use_sim3", "--max_loops", "7",
            "--log_path", poses_out,
        ]), logfile)
    finally:
        restore_fn()

    if os.path.isfile(poses_out):
        with open(poses_out) as f:
            n = sum(1 for _ in f)
        log(f"STEP 6 done. poses.txt: {n} lines", logfile)
    else:
        log("WARNING: poses.txt not found", logfile)

    log(f"CLIP COMPLETE: {name}", logfile)
    log("="*60, logfile)
    return True

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=str, default=None,
                    help="Run only this clip by folder name e.g. 239eef93")
    ap.add_argument("--only-convert", action="store_true",
                    help="Only run step 0 (equirect->perspective) without SLAM")
    args = ap.parse_args()

    os.makedirs(OUT_BASE, exist_ok=True)

    if args.clip:
        clips = [(c, n, d) for c, n, d in CLIPS if c == args.clip]
    else:
        clips = CLIPS

    for clip_id, name, desc in clips:
        if args.only_convert:
            root = os.path.join(OUT_BASE, name)
            os.makedirs(root, exist_ok=True)
            seq = convert_clip_to_perspective(clip_id, name)
            print(f"Converted {clip_id} -> {seq}")
            # show one sample
            imgs = sorted(os.listdir(seq))
            if imgs:
                import cv2
                img = cv2.imread(os.path.join(seq, imgs[0]))
                print(f"Sample frame shape: {img.shape}")
            continue
        try:
            ok = run_clip(clip_id, name, desc)
            print(f"{'✓' if ok else '✗'} {name}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR {name}: {e}")
            import traceback; traceback.print_exc()

    print("\nAll done.")