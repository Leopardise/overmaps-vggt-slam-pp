#!/usr/bin/env python3
"""
Compare two TUM-style trajectories (ground-truth **ref.txt** and estimated **est.txt**).

Format (TUM):
    timestamp tx ty tz qx qy qz qw

Logic (as requested)
--------------------
1) Round each timestamp to the nearest whole second, then shift so the
   first timestamp becomes Δt = 0 s.
2) Keep only the frames whose rounded Δt values occur in *both* logs.
3) If ≥ 3 matched frames exist:
       – Estimate a similarity transform (scale s, rotation R, translation t)
         that maps **est → ref** via Umeyama.
   Else (1–2 matches):
       – Align the first pose only (pure translation, no scale, no rotation).
4) Apply that transform to *every* pose in **est.txt** (ref is untouched).
5) Print alignment error metrics and plot both trajectories in 3D.

CLI
---
python compare_tum.py --ref ref.txt --est est.txt [--no-scale] [--no-plot]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def parse_args():
    ap = argparse.ArgumentParser(description="TUM trajectory comparison (est → ref alignment)")
    ap.add_argument("--ref", required=True, help="Path to ground-truth ref.txt")
    ap.add_argument("--est", required=True, help="Path to estimated est.txt")
    ap.add_argument("--no-scale", action="store_true", help="Disable scale estimation (rigid only)")
    ap.add_argument("--no-plot", action="store_true", help="Skip the 3D plot")
    return ap.parse_args()

def load_tum(path):
    """
    Return:
        dict: {rounded_rel_time_int : pose[tx ty tz qx qy qz qw]}
    Details:
        - Ignores comment/blank lines.
        - Keeps the last pose for a duplicate rounded second.
    """
    rows = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) != 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            rows.append((ts, np.array([tx, ty, tz, qx, qy, qz, qw], dtype=float)))

    if not rows:
        raise RuntimeError(f"No valid poses in {path}")

    t0_int = int(round(rows[0][0]))
    out = {}
    for ts, pose in rows:
        key = int(round(ts)) - t0_int
        out[key] = pose
    return out

def umeyama(src_xyz, dst_xyz, with_scale=True):
    """
    Umeyama similarity transform (1991): maps x_src → s*R*x_src + t ≈ x_dst
    Returns (s, R3x3, t3,)
    """
    n = src_xyz.shape[0]
    mu_s = src_xyz.mean(0)
    mu_d = dst_xyz.mean(0)
    X = src_xyz - mu_s
    Y = dst_xyz - mu_d
    Sigma = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(Sigma)
    Sgn = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sgn[2, 2] = -1
    R_opt = U @ Sgn @ Vt
    s_opt = 1.0
    if with_scale:
        var = (X * X).sum() / n
        s_opt = np.trace(np.diag(D) @ Sgn) / var
    t_opt = mu_d - s_opt * (R_opt @ mu_s)
    return s_opt, R_opt, t_opt

def set_equal_3d(ax, pts_a, pts_b):
    """Equal aspect for 3D by setting box_aspect to the combined extents."""
    pts = np.vstack([pts_a, pts_b])
    mins = pts.min(0)
    maxs = pts.max(0)
    spans = np.maximum(maxs - mins, 1e-9)
    ax.set_box_aspect(spans)

def main():
    args = parse_args()
    ref_dict = load_tum(args.ref)
    est_dict = load_tum(args.est)

    common = sorted(set(ref_dict) & set(est_dict))
    if not common:
        raise RuntimeError("No rounded-second timestamps in common.")

    ref_xyz = np.vstack([ref_dict[k][:3] for k in common])
    est_xyz = np.vstack([est_dict[k][:3] for k in common])

    if len(common) >= 3:
        s, R_opt, t_opt = umeyama(est_xyz, ref_xyz, with_scale=not args.no_scale)
        print(f"[Umeyama] scale s       = {s:.9f}")
        print(f"[Umeyama] translation t = {t_opt}")
        print(f"[Umeyama] rotation R =\n{R_opt}")
    else:
        s = 1.0
        R_opt = np.eye(3)
        t_opt = ref_xyz[0] - est_xyz[0]
        print("<Fallback> Only 1–2 matches → first-pose translation only.")

    # Errors over matched subset
    est_xyz_aligned_sub = (s * (R_opt @ est_xyz.T).T) + t_opt
    err = np.linalg.norm(est_xyz_aligned_sub - ref_xyz, axis=1)
    stats = {
        "count" : err.size,
        "max"   : float(np.max(err)),
        "mean"  : float(np.mean(err)),
        "median": float(np.median(err)),
        "min"   : float(np.min(err)),
        "rmse"  : float(np.sqrt(np.mean(err**2))),
        "sse"   : float(np.sum(err**2)),
        "std"   : float(np.std(err)),
    }
    print("\nAlignment error statistics (meters):")
    for k in ["count","min","max","mean","median","std","rmse","sse"]:
        print(f"{k:>6s}: {stats[k]:.6f}" if k!="count" else f"{k:>6s}: {int(stats[k])}")

    # Transform full estimated trajectory
    est_full = np.array([p for _, p in sorted(est_dict.items())])
    est_full_xyz = est_full[:, :3]
    est_full_q   = est_full[:, 3:]
    est_xyz_aligned = (s * (R_opt @ est_full_xyz.T).T) + t_opt
    R_obj = R.from_matrix(R_opt)
    est_q_aligned = (R_obj * R.from_quat(est_full_q)).as_quat()  # (x,y,z,w)

    # Prepare full ref for plot
    ref_full = np.array([p for _, p in sorted(ref_dict.items())])
    ref_full_xyz = ref_full[:, :3]

    if not args.no_plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(ref_full_xyz[:,0], ref_full_xyz[:,1], ref_full_xyz[:,2],
                label="Ground Truth (ref)", linewidth=2)
        ax.plot(est_xyz_aligned[:,0], est_xyz_aligned[:,1], est_xyz_aligned[:,2],
                label="Estimated aligned (est)", linewidth=2)
        txt = "\n".join([
            f"count: {int(stats['count'])}",
            f"min  : {stats['min']:.3f}",
            f"max  : {stats['max']:.3f}",
            f"mean : {stats['mean']:.3f}",
            f"median:{stats['median']:.3f}",
            f"std  : {stats['std']:.3f}",
            f"rmse : {stats['rmse']:.3f}",
        ])
        ax.text2D(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
        ax.set_title("Trajectory Comparison  (est → ref)")
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        set_equal_3d(ax, ref_full_xyz, est_xyz_aligned)
        ax.legend(); ax.grid(True); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
