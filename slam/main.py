import os, glob, argparse, cv2, numpy as np, torch, json, tempfile, time
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.global_dem_tiled import render_global_dem_tiled

# ---------- atomic writers ----------
def write_json_atomic(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # POSIX atomic

def save_npy_atomic(path: str, arr: np.ndarray) -> None:
    # write into same dir so rename is atomic across filesystems
    d = os.path.dirname(path); Path(d).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".tmp_", suffix=".npy", dir=d, delete=False) as tf:
        np.save(tf, arr)
        tf.flush(); os.fsync(tf.fileno())
        tmp = tf.name
    os.replace(tmp, path)

# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser("VGGT-SLAM++: VGGT front-end + DEM covisibility back-end")
    # SLAM (paper defaults: Sim(3), w=32, w_loops=0, τ_disparity=40)
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--use_sim3", action="store_true", default=True,
                   help="Sim(3) odometry / back-end (paper default; always on)")
    p.add_argument("--no_sim3", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--submap_size", type=int, default=32)
    p.add_argument("--overlapping_window_size", type=int, default=1)
    p.add_argument("--downsample_factor", type=int, default=1)
    p.add_argument("--min_disparity", type=float, default=40.0)
    p.add_argument("--use_point_map", action="store_true")
    p.add_argument("--conf_threshold", type=float, default=25.0)
    p.add_argument("--vis_stride", type=int, default=1)
    p.add_argument("--vis_point_size", type=float, default=0.003)
    p.add_argument("--depth_min", type=float, default=-60.0)
    p.add_argument("--depth_max", type=float, default=100.0)
    p.add_argument("--max_loops", type=int, default=0)
    p.add_argument("--vis_map", action="store_true")
    p.add_argument("--vis_flow", action="store_true")
    p.add_argument("--log_results", action="store_true")
    p.add_argument("--skip_dense_log", action="store_true")
    p.add_argument("--log_path", type=str, default="poses.txt")
    p.add_argument("--plot_focal_lengths", action="store_true")

    # DEM (global)
    p.add_argument("--make_global_dem_tiled", action="store_true")
    p.add_argument("--global_dem_out_dir", type=str, default="outputs/run")
    p.add_argument("--global_target_px", type=int, default=90000)
    p.add_argument("--global_tile_px", type=int, default=4096)
    p.add_argument("--global_kernel_px", type=float, default=1.2)
    p.add_argument("--global_reducer", type=str, choices=["mean","max","softmax"], default="softmax")
    p.add_argument("--global_softmax_tau", type=float, default=0.02)
    p.add_argument("--global_radius_keep_pct", type=float, default=100.0)
    p.add_argument("--global_clip_lo", type=float, default=0.0)
    p.add_argument("--global_clip_hi", type=float, default=100.0)

    # Visual controls (parity with renderer)
    p.add_argument("--global_cycle_m", type=float, default=0.001)
    p.add_argument("--global_edge_strength", type=float, default=0.95)
    p.add_argument("--global_shade_strength", type=float, default=0.70)
    p.add_argument("--global_dark_level", type=float, default=0.35)
    p.add_argument("--global_unsharp_radius", type=float, default=1.8)
    p.add_argument("--global_unsharp_amount", type=float, default=1.4)
    p.add_argument("--global_clahe_clip", type=float, default=3.0)
    p.add_argument("--global_clahe_grid", type=int, default=8)

    # Submap DUMP for backend watcher
    p.add_argument("--dump_submaps", action="store_true",
                   help="Write per-submap points to <out>/submaps/sm_xxxxx/")
    p.add_argument("--submaps_out_root", type=str, default="",
                   help="Override submaps output root (default == global_dem_out_dir)")
    p.add_argument("--render_initial_after_submaps", type=int, default=0,
                   help="If >0, render a first global DEM after N dumped submaps so backend can start")

    # Backbone / ckpt
    p.add_argument("--vggt_ckpt", type=str, default="",
                   help="VGGT ckpt path OR MapAnything HF id if --use_mapanything is set")
    p.add_argument("--use_mapanything", action="store_true",
                   help="If set, load MapAnything instead of VGGT (minimal adapter in Solver.run_predictions)")
        # ---------- Evaluation ----------
    p.add_argument("--chamfer", action="store_true",
                help="Compute Chamfer distance against GT mesh")
    p.add_argument("--gt_mesh", type=str, default="",
                help="Path to GT mesh.ply (Replica / ScanNet++)")
    p.add_argument("--eval_sample_pts", type=int, default=1000000,
                help="Number of points to sample from GT mesh")
    p.add_argument("--eval_icp_thresh", type=float, default=0.05,
                help="ICP max correspondence distance (meters)")
    p.add_argument("--external_loops_csv", type=str, default="",
                   help="AnyLoc/DEM loop_votes.csv from the back-end (paper Sec. 3)")
    return p.parse_args()

def load_model(device: str, ckpt_path: str = "", use_mapanything: bool = False):
    """
    Minimal switch:
      - use_mapanything == False -> original VGGT
      - use_mapanything == True  -> MapAnything.from_pretrained(...)
    """
    if use_mapanything:
        print("Initializing MapAnything backbone…")
        from mapanything.models import MapAnything  # lazy import
        model_name = ckpt_path if ckpt_path else "facebook/map-anything"
        model = MapAnything.from_pretrained(model_name)
        model.eval()
        return model.to(device)
    else:
        print("Initializing VGGT backbone…")
        from vggt.models.vggt import VGGT  # original import
        model = VGGT()
        if ckpt_path and os.path.isfile(ckpt_path):
            sd = torch.load(ckpt_path, map_location=device)
        else:
            if not ckpt_path:
                print("No --vggt_ckpt provided; downloading VGGT-1B…")
                url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
                sd = torch.hub.load_state_dict_from_url(url, map_location=device)
            else:
                raise FileNotFoundError(f"VGGT checkpoint not found: {ckpt_path}")
        model.load_state_dict(sd); model.eval()
        return model.to(device)

# ---------- submap export ----------
def _try_get_submap_points_world(sm, stride: int = 1) -> np.ndarray:
    for name in ["get_points_in_world_frame", "get_points_world", "get_points"]:
        fn = getattr(sm, name, None)
        if callable(fn):
            try:
                P = fn(stride=stride) if "stride" in fn.__code__.co_varnames else fn()
                P = np.asarray(P, dtype=np.float32).reshape(-1, 3)
                if P.size:
                    return P
            except Exception:
                pass
    return np.zeros((0,3), np.float32)

def export_this_submap(sm, out_root: str,
                       stride_pts: int,
                       clip_lo: float, clip_hi: float,
                       kernel_px: float, reducer: str, softmax_tau: float):
    """
    Dump exactly THIS Submap's points to disk, freezing its content.
    """
    # pull points from this submap object (already in WORLD frame)
    P = _try_get_submap_points_world(sm, stride=stride_pts).astype(np.float32)
    sm_id = int(getattr(sm, "submap_id", sm.get_id()))

    sm_dir = os.path.join(out_root, "submaps", f"sm_{sm_id:05d}")
    Path(sm_dir).mkdir(parents=True, exist_ok=True)

    # write points atomically
    save_npy_atomic(os.path.join(sm_dir, "points_world.npy"), P)

    # small meta (for sanity checks)
    meta = {
        "sm_id": sm_id,
        "num_points": int(P.shape[0]),
        "clip_lo": float(clip_lo),
        "clip_hi": float(clip_hi),
        "kernel_px": float(kernel_px),
        "reducer": reducer,
        "softmax_tau": float(softmax_tau),
        "timestamp": int(time.time()),
    }
    write_json_atomic(os.path.join(sm_dir, "meta.json"), meta)

    # READY marker lets the backend watcher know it can chip
    Path(os.path.join(sm_dir, "READY")).write_text("ok\n")

    # extra logs to catch “all submaps identical” bugs quickly
    if P.size > 0:
        xyz_min = P.min(axis=0).tolist()
        xyz_max = P.max(axis=0).tolist()
    else:
        xyz_min = xyz_max = [0.0, 0.0, 0.0]
    print(f"[frontend] dumped sm_{sm_id:05d}: {P.shape[0]} pts | "
          f"xyz_min={np.round(xyz_min,3)} xyz_max={np.round(xyz_max,3)} "
          f"→ {sm_dir}")

    return sm_dir

# ---------- main ----------
def main():
    args = get_args()
    if getattr(args, "no_sim3", False):
        args.use_sim3 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False,
        vis_stride=args.vis_stride,
        vis_point_size=args.vis_point_size,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    model = load_model(device, args.vggt_ckpt, args.use_mapanything)

    print(f"Loading images from {args.image_folder} …")
    names = [f for f in glob.glob(os.path.join(args.image_folder, "*"))
             if "depth" not in os.path.basename(f).lower()
             and "txt" not in os.path.basename(f).lower()
             and "db" not in os.path.basename(f).lower()]
    names = utils.sort_images_by_number(names)
    names = utils.downsample_images(names, args.downsample_factor)
    print(f"Found {len(names)} images")

    # Where submaps will be written
    out_root = args.submaps_out_root if args.submaps_out_root else args.global_dem_out_dir
    Path(os.path.join(out_root, "submaps")).mkdir(parents=True, exist_ok=True)

    subset, focal_history = [], []
    dumped = 0
    did_initial_render = False

    for name in tqdm(names):
        img = cv2.imread(name)
        if solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow):
            subset.append(name)

        end_of_window = (len(subset) == args.submap_size + args.overlapping_window_size) or (name == names[-1])
        if end_of_window and subset:
            with torch.no_grad():
                pred = solver.run_predictions(subset, model, args.max_loops)
            focal_history.append(pred["intrinsic"][:, 0, 0])

            # 1) Update SLAM with this submap so it has correct world frame and homography
            solver.add_points(pred)
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            # 2) Dump exactly this submap to disk for the backend watcher
            if args.dump_submaps:
                sm_obj = solver.current_working_submap
                export_this_submap(
                    sm=sm_obj,
                    out_root=out_root,
                    stride_pts=max(1, args.vis_stride * 2),
                    clip_lo=args.global_clip_lo,
                    clip_hi=args.global_clip_hi,
                    kernel_px=args.global_kernel_px,
                    reducer=args.global_reducer,
                    softmax_tau=args.global_softmax_tau,
                )
                dumped += 1

                # Optional: early global DEM so backend has index.json quickly
                if (args.make_global_dem_tiled and
                    args.render_initial_after_submaps > 0 and
                    not did_initial_render and
                    dumped >= args.render_initial_after_submaps):
                    print(f"[frontend] initial global DEM after {dumped} dumped submaps …")
                    render_global_dem_tiled(
                        graph_map=solver.get_graph_map(),
                        out_dir=args.global_dem_out_dir,
                        radius_keep_pct=args.global_radius_keep_pct,
                        clip_lo=args.global_clip_lo, clip_hi=args.global_clip_hi,
                        target_px_long=args.global_target_px,
                        tile_px=args.global_tile_px,
                        kernel_px=args.global_kernel_px,
                        reducer=args.global_reducer,
                        softmax_tau=args.global_softmax_tau,
                        cycle_meters=args.global_cycle_m,
                        edge_strength=args.global_edge_strength,
                        shade_strength=args.global_shade_strength,
                        dark_level=args.global_dark_level,
                        unsharp_radius_px=args.global_unsharp_radius,
                        unsharp_amount=args.global_unsharp_amount,
                        clahe_clip=args.global_clahe_clip,
                        clahe_grid=args.global_clahe_grid,
                        stride=max(1, args.vis_stride * 2),
                    )
                    did_initial_render = True

            if args.vis_map:
                (solver.update_all_submap_vis()
                 if len(pred["detected_loops"]) > 0 else solver.update_latest_submap_vis())

            # keep overlap for next submap
            subset = subset[-args.overlapping_window_size:]

    print("Total submaps:", solver.map.get_num_submaps())
    print("Total loop closures:", solver.graph.get_num_loops())

    # Final global DEM render (full sequence)
    if args.make_global_dem_tiled:
        render_global_dem_tiled(
            graph_map=solver.get_graph_map(),
            out_dir=args.global_dem_out_dir,
            radius_keep_pct=args.global_radius_keep_pct,
            clip_lo=args.global_clip_lo, clip_hi=args.global_clip_hi,
            target_px_long=args.global_target_px,
            tile_px=args.global_tile_px,
            kernel_px=args.global_kernel_px,
            reducer=args.global_reducer,
            softmax_tau=args.global_softmax_tau,
            cycle_meters=args.global_cycle_m,
            edge_strength=args.global_edge_strength,
            shade_strength=args.global_shade_strength,
            dark_level=args.global_dark_level,
            unsharp_radius_px=args.global_unsharp_radius,
            unsharp_amount=args.global_unsharp_amount,
            clahe_clip=args.global_clahe_clip,
            clahe_grid=args.global_clahe_grid,
            stride=max(1, args.vis_stride * 2),
        )

    if args.external_loops_csv:
        added = solver.apply_external_loops_if_any(args.external_loops_csv, optimize_each=True)
        print(f"[backend] applied {added} DEM/AnyLoc loop edges")

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path)
        if not args.skip_dense_log:
            solver.map.save_framewise_pointclouds(args.log_path.replace(".txt", "_logs"))
            
    # ---------- Chamfer evaluation ----------
        
    if args.chamfer:
        if not (args.gt_mesh and os.path.isfile(args.gt_mesh)):
            raise FileNotFoundError("--gt_mesh must be provided for --chamfer and must exist")

        try:
            from vggt_slam.eval_reconstruction import (
                load_all_submap_points,
                sample_gt_mesh,
                compute_chamfer,
            )
        except Exception as e:
            raise ImportError(
                "Chamfer eval requested (--chamfer), but vggt_slam.eval_reconstruction "
                "is missing or not importable in this checkout.\n"
                "Either add vggt_slam/eval_reconstruction.py, or disable --chamfer.\n"
                f"Original error: {repr(e)}"
            )

        print("[eval] loading predicted global point cloud …")
        submaps_root = os.path.join(out_root, "submaps")
        pred_pts = load_all_submap_points(submaps_root)

        print("[eval] sampling GT mesh …")
        gt_pcd = sample_gt_mesh(args.gt_mesh, args.eval_sample_pts)

        print("[eval] computing Chamfer distance …")
        metrics = compute_chamfer(
            pred_pts,
            gt_pcd,
            icp_thresh=args.eval_icp_thresh,
        )

        metrics_path = os.path.join(out_root, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("[eval] results:", metrics)
        print("[eval] saved to:", metrics_path)

            
    if args.plot_focal_lengths and len(focal_history) > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(focal_history)))
        plt.figure(figsize=(8, 5))
        for i, vals in enumerate(focal_history):
            x = [i] * len(vals)
            plt.scatter(x, vals, color=colors[i], s=6)
        plt.grid(True); plt.xlabel("poses"); plt.ylabel("focal"); plt.show()

if __name__ == "__main__":
    main()
