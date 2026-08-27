#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------- parsing -------------------------------------
def load_tum_xyz(path):
    rows = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                tx, ty, tz = map(float, parts[1:4])
                rows.append([tx, ty, tz])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No valid xyz rows in {path}")
    return np.asarray(rows, dtype=float)

# ------------------------------ geometry -------------------------------------
def cum_arclen(xyz):
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s

def compute_path_length(xyz):
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))

def resample_by_arclen(xyz, n_samples=500):
    if xyz.shape[0] == 1:
        return np.repeat(xyz, n_samples, axis=0)
    s = cum_arclen(xyz)
    if s[-1] < 1e-12:
        return np.repeat(xyz[:1], n_samples, axis=0)
    targ = np.linspace(0.0, s[-1], n_samples)
    out = np.empty((n_samples, 3), dtype=float)
    j = 0
    for i, si in enumerate(targ):
        while j+1 < s.size and s[j+1] < si:
            j += 1
        if j+1 == s.size:
            out[i] = xyz[-1]
        else:
            t = 0.0 if s[j+1] == s[j] else (si - s[j]) / (s[j+1] - s[j])
            out[i] = (1.0 - t) * xyz[j] + t * xyz[j+1]
    return out

def umeyama(src_xyz, dst_xyz, with_scale=True):
    assert src_xyz.shape == dst_xyz.shape and src_xyz.shape[1] == 3
    n = src_xyz.shape[0]
    mu_s = src_xyz.mean(axis=0)
    mu_d = dst_xyz.mean(axis=0)
    X = src_xyz - mu_s
    Y = dst_xyz - mu_d
    Sigma = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(Sigma)
    Sgn = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sgn[2, 2] = -1
    R = U @ Sgn @ Vt
    s = 1.0
    if with_scale:
        var = (X * X).sum() / n
        s = np.trace(np.diag(D) @ Sgn) / var if var > 0 else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t

# ------------------------------ metrics --------------------------------------
def shape_metrics(A, B):
    d = np.linalg.norm(A - B, axis=1)
    return {
        "count": int(d.size),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "rmse": float(np.sqrt(np.mean(d**2))),
        "std": float(np.std(d)),
        "chamfer_mean": float(np.mean(d)),
        "hausdorff": float(np.max(d)),
    }

# ------------------------------ plotting -------------------------------------
def equal_3d(ax, A, B):
    pts = np.vstack([A, B])
    mins = pts.min(0); maxs = pts.max(0)
    span = np.maximum(maxs - mins, 1e-9)
    ax.set_box_aspect(span)

# ------------------------------ main -----------------------------------------
def main():
    ap = argparse.ArgumentParser(description="REF → EST shape-based ATE with scale-adjusted errors")
    ap.add_argument("--ref", required=True, help="Ground truth file")
    ap.add_argument("--est", required=True, help="Estimated trajectory")
    ap.add_argument("--resample", type=int, default=500, help="Number of resampled points")
    ap.add_argument("--no-plot", action="store_true", help="Skip plot")
    args = ap.parse_args()

    ref_xyz_raw = load_tum_xyz(args.ref)
    est_xyz_raw = load_tum_xyz(args.est)

    ref_xyz0 = ref_xyz_raw - ref_xyz_raw[0]
    est_xyz0 = est_xyz_raw - est_xyz_raw[0]

    N = max(2, args.resample)
    ref_s = resample_by_arclen(ref_xyz0, N)
    est_s = resample_by_arclen(est_xyz0, N)

    # Align ref → est
    s, R, t = umeyama(ref_s, est_s, with_scale=True)
    ref_aligned = (s * (R @ ref_s.T).T) + t

    stats = shape_metrics(ref_aligned, est_s)

    # Compute path lengths
    gt_len = compute_path_length(ref_xyz_raw)
    est_len = compute_path_length(est_xyz_raw)
    length_ratio = gt_len / est_len if est_len > 1e-6 else 1.0

    # Scale errors
    scaled_stats = {k: (v * length_ratio if isinstance(v, float) else v) for k, v in stats.items()}

    print("\n== REF → EST shape alignment ==")
    print(f"Transform: scale={s:.9f}")
    print("Rotation:\n", R)
    print("Translation:", t)
    print(f"\nGT path length : {gt_len:.3f} m")
    print(f"EST path length: {est_len:.3f} m")
    print(f"Length ratio   : {length_ratio:.6f}")

    print("\nError statistics (meters):")
    for k in ["count","min","max","mean","median","std","rmse","chamfer_mean","hausdorff"]:
        v = stats[k]
        print(f"{k:>13s}: {v:.6f}" if isinstance(v, float) else f"{k:>13s}: {v}")

    print("\nLength-scaled error statistics:")
    for k in ["min","max","mean","median","std","rmse","chamfer_mean","hausdorff"]:
        v = scaled_stats[k]
        print(f"{k:>13s}: {v:.6f}")

    # Plot
    if not args.no_plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(ref_aligned[:,0], ref_aligned[:,1], ref_aligned[:,2], label="REF aligned", linewidth=2)
        ax.plot(est_s[:,0], est_s[:,1], est_s[:,2], label="EST", linewidth=2)
        ax.set_title("Shape ATE (REF → EST, Sim(3))")
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        equal_3d(ax, ref_aligned, est_s)
        ax.legend(); ax.grid(True); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
