#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose-only refinement from chip↔tile matches (2D-3D via DEM), with ORB-SLAM3-style robustness:
AP3P -> EPNP -> IPPE (planar) -> Homography (planar) decomposition, then robust g2o pose-only BA.

CLI example:
  python vggt_slam/tools/pose_only_from_matches.py \
    --root outputs/00 \
    --io-dir outputs/00/anyloc_io/sm_00001 \
    --matches outputs/00/anyloc_io/sm_00001/matches.csv \
    --K 718.856,0,607.1928,0,718.856,185.2157,0,0,1 \
    --dist 0,0,0,0,0 \
    --max-pairs 10000 \
    --save-pose outputs/00/submaps/sm_00001/pose_refined_se3.npy
"""

import os, json, argparse, csv
import numpy as np
import cv2

# ----------------- load global contract -----------------

def load_index(root):
    idx = json.load(open(os.path.join(root, "index.json"), "r"))
    return {
        "tile_px": int(idx["grid_size_px"]),
        "mpp": float(idx["target_mpp"]),
        "nx": int(idx["nx"]),
        "ny": int(idx["ny"]),
        "bbox": tuple(idx["bbox_global"]),
        "U": np.asarray(idx["plane_U"], np.float64),
        "V": np.asarray(idx["plane_V"], np.float64),
        "N": np.asarray(idx["plane_N"], np.float64),
        "d0": float(idx["plane_d"]),
        "R2": np.asarray(idx["pca_R2"], np.float64),
        "mu": np.asarray(idx["pca_mu_xy"], np.float64),
    }

def tile_id_from_path(png_path):
    stem = os.path.splitext(os.path.basename(png_path))[0]
    return int(stem.split("_")[-1])

def tile_bbox_from_id(meta, tid):
    nx = meta["nx"]; tile_px = meta["tile_px"]; mpp = meta["mpp"]
    ty, tx = divmod(tid, nx)
    x0g, y0g = meta["bbox"][0], meta["bbox"][1]
    x0 = x0g + tx * tile_px * mpp
    y0 = y0g + ty * tile_px * mpp
    x1 = x0 + tile_px * mpp
    y1 = y0 + tile_px * mpp
    return (x0, y0, x1, y1)

def tile_pixel_to_world(meta, tid, px, py, dem_val):
    tb = tile_bbox_from_id(meta, tid)
    mpp = meta["mpp"]
    x = tb[0] + (px + 0.5) * mpp
    y = tb[1] + (py + 0.5) * mpp
    U, V, N, d0 = meta["U"], meta["V"], meta["N"], meta["d0"]
    X = x * U + y * V + (dem_val - d0) * N
    return X.astype(np.float64)

# ----------------- CSV + SIFT matching -----------------

def read_matches(csv_path, keep_top1=True):
    rows = []
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        need = {"query_path","db_path"}
        if not need.issubset(set(rdr.fieldnames or [])):
            raise RuntimeError(f"Bad CSV schema in {csv_path}; need query_path,db_path[,score]")
        for r in rdr:
            s = float(r.get("score", "0"))
            rows.append((r["query_path"], r["db_path"], s))
    if keep_top1:
        best = {}
        for q, d, s in rows:
            if (q not in best) or (s > best[q][1]):
                best[q] = (d, s)
        rows = [(q, best[q][0], best[q][1]) for q in best]
    return rows

def detect_and_match_sift(q_img, d_img, max_pairs=10000, ratio=0.8):
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT (contrib) required")
    sift = cv2.SIFT_create(nfeatures=max_pairs)
    kq, dq = sift.detectAndCompute(q_img, None)
    kd, dd = sift.detectAndCompute(d_img, None)
    if dq is None or dd is None or len(kq) < 8 or len(kd) < 8:
        return np.empty((0,2)), np.empty((0,2))
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    m = bf.knnMatch(dq, dd, k=2)
    good = []
    for pair in m:
        if len(pair) < 2: continue
        a, b = pair
        if a.distance < ratio * b.distance:
            good.append(a)
    good = sorted(good, key=lambda x: x.distance)[:max_pairs]
    if not good:
        return np.empty((0,2)), np.empty((0,2))
    q_pts = np.float32([kq[g.queryIdx].pt for g in good])
    t_pts = np.float32([kd[g.trainIdx].pt for g in good])
    return q_pts, t_pts

def local_flat_mask(dem, px, py, win=3, std_thresh=1e-6):
    h, w = dem.shape
    px = np.clip(px, 0, w-1); py = np.clip(py, 0, h-1)
    half = win // 2
    ok = []
    for x, y in zip(px, py):
        x0, x1 = max(0, x-half), min(w, x+half+1)
        y0, y1 = max(0, y-half), min(h, y+half+1)
        patch = dem[y0:y1, x0:x1]
        ok.append(patch.std() > std_thresh)
    return np.array(ok, dtype=bool)

def grid_thin(uv, grid=24, W=None, H=None):
    if len(uv) == 0: return np.empty((0,), int)
    if W is None: W = int(np.ceil(uv[:,0].max())) + 1
    if H is None: H = int(np.ceil(uv[:,1].max())) + 1
    gx = np.clip((uv[:,0] / max(1, grid)).astype(int), 0, W)
    gy = np.clip((uv[:,1] / max(1, grid)).astype(int), 0, H)
    cells = gx + 100000*gy
    seen = {}
    keep = []
    for i, c in enumerate(cells):
        if c not in seen:
            seen[c] = i
            keep.append(i)
    return np.array(keep, dtype=int)

# ----------------- correspondence gather -----------------

def gather_correspondences(root, io_dir, matches_csv, max_pairs):
    meta = load_index(root)
    tiles_dir = os.path.join(root, "tiles")
    pairs = read_matches(matches_csv, keep_top1=True)

    Q2d, W3d, AB = [], [], []  # AB are plane coords (a,b,1) used by homography/IPPE

    for q_path, d_path, _ in pairs:
        chip = cv2.imread(q_path, cv2.IMREAD_GRAYSCALE)
        tile = cv2.imread(d_path, cv2.IMREAD_GRAYSCALE)
        if chip is None or tile is None: continue

        q_uv, t_uv = detect_and_match_sift(chip, tile, max_pairs=max_pairs, ratio=0.8)
        if len(q_uv) < 8: continue

        tid = tile_id_from_path(d_path)
        dem_npy = os.path.join(tiles_dir, f"tile_{tid:05d}.npy")
        if not os.path.isfile(dem_npy): continue
        dem = np.load(dem_npy).astype(np.float32)

        Ht, Wt = dem.shape
        px = np.clip(np.round(t_uv[:,0]).astype(np.int32), 0, Wt-1)
        py = np.clip(np.round(t_uv[:,1]).astype(np.int32), 0, Ht-1)
        z  = dem[py, px]

        valid = (z != 0.0)
        valid &= local_flat_mask(dem, px, py, win=3, std_thresh=1e-6)
        if not np.any(valid): continue

        q_uv = q_uv[valid]; px = px[valid]; py = py[valid]; z = z[valid]
        keep_idx = grid_thin(q_uv, grid=24, W=chip.shape[1], H=chip.shape[0])
        q_uv = q_uv[keep_idx]; px = px[keep_idx]; py = py[keep_idx]; z = z[keep_idx]
        if len(q_uv) < 6: continue

        # world & plane coords for each match
        U, V, N, d0 = meta["U"], meta["V"], meta["N"], meta["d0"]
        for (u,v), ix, iy, zz in zip(q_uv, px, py, z):
            Xw = tile_pixel_to_world(meta, tid, int(ix), int(iy), float(zz))
            a = float(Xw @ U); b = float(Xw @ V)   # plane coords (Z=0)
            Q2d.append([u, v]); W3d.append(Xw); AB.append([a, b, 1.0])

    return np.asarray(Q2d, np.float64), np.asarray(W3d, np.float64), np.asarray(AB, np.float64), meta

# ----------------- initializers -----------------

def try_pnp_ransac(W3d, Q2d, K, dist, flags, thr_px, iters=4000):
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        W3d, Q2d, K, dist,
        flags=flags,
        reprojectionError=float(thr_px),
        iterationsCount=int(iters),
        confidence=0.999
    )
    if not ok or inl is None or len(inl) < 6:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(W3d[inl], Q2d[inl], K, dist, rvec, tvec)
    R,_ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = tvec.reshape(3)
    return T, inl.reshape(-1)

def try_ippe(Q2d, W3d, K, dist, meta):
    if len(Q2d) < 4: return None
    U, V, N, d0 = meta["U"], meta["V"], meta["N"], meta["d0"]
    obj = np.stack([W3d @ U, W3d @ V, np.zeros(len(W3d))], axis=1).astype(np.float64)
    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, Q2d, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not retval or rvecs is None or len(rvecs) == 0:
        return None

    # world <- object (plane)
    T_wo = np.eye(4); T_wo[:3,:3] = np.stack([U, V, N], axis=1); T_wo[:3,3] = -d0 * N
    best = None; best_err = 1e18; best_inl = None
    for rvec, tvec in zip(rvecs, tvecs):
        R,_ = cv2.Rodrigues(rvec)
        T_co = np.eye(4); T_co[:3,:3] = R; T_co[:3,3] = tvec.reshape(3)
        T_oc = np.eye(4); T_oc[:3,:3] = R.T; T_oc[:3,3] = -R.T @ tvec.reshape(3)
        T_wc = T_wo @ T_oc

        # score by reprojection
        Rw = T_wc[:3,:3]; tw = T_wc[:3,3].reshape(3,1)
        Xc = (Rw @ W3d.T + tw).T
        if np.count_nonzero(Xc[:,2] > 1e-6) < len(Xc) * 0.6:  # cheirality
            continue
        uv_hat = (K @ (Xc.T / Xc[:,2]).T[...,None]).squeeze(-1)[:, :2]
        e = np.linalg.norm(Q2d - uv_hat, axis=1)
        inl = np.where(e < 8.0)[0]
        err = np.median(e[inl]) if len(inl) else 1e18
        if err < best_err and len(inl) >= 6:
            best_err, best, best_inl = err, T_wc, inl
    if best is None:
        return None
    return best, best_inl

def try_homography(Q2d, AB, K):
    """Planar initializer like ORB: H (plane->image) via RANSAC, then decompose pose."""
    if len(Q2d) < 8: return None
    # AB are metric plane coords (a,b,1). We map (a,b,1) -> image (u,v,1)
    H, mask = cv2.findHomography(AB[:, :2], Q2d, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999)
    if H is None or mask is None or np.count_nonzero(mask) < 6:
        return None
    # K^{-1} H = [h1 h2 h3], up to scale; R = [r1 r2 r3], t = h3 / s
    Kinv = np.linalg.inv(K)
    B = Kinv @ H
    b1 = B[:,0]; b2 = B[:,1]; b3 = B[:,2]
    # enforce orthonormality like ORB: s = 1/avg(||b1||,||b2||)
    s = 1.0 / max(1e-12, 0.5 * (np.linalg.norm(b1) + np.linalg.norm(b2)))
    r1 = s * b1; r2 = s * b2; r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    # project to SO(3)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0: R[:,2] *= -1
    t = s * b3
    # build T
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3]  = t
    # inlier mask from homography already available; create indices
    inl = np.where(mask.ravel().astype(bool))[0]
    return T, inl

# ----------------- g2o pose-only BA -----------------

def g2o_refine_pose(T_init, Q2d, W3d, K, iters=30):
    try:
        import pyg2o as g2o
    except Exception as e:
        raise RuntimeError("pyg2o required (pip install pyg2o or build)") from e

    fx, fy = float(K[0,0]), float(K[1,1]); cx, cy = float(K[0,2]), float(K[1,2])
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(solver))

    cam = g2o.CameraParameters(fx, g2o.Vector2(cx, cy), 0)
    cam.set_id(0); optimizer.add_parameter(cam)

    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_id(0)
    v_se3.set_estimate(g2o.SE3Quat(T_init[:3,:3], T_init[:3,3]))
    optimizer.add_vertex(v_se3)

    huber = g2o.RobustKernelHuber()
    # ORB uses Tukey/Huber; Huber is fine here
    for i, (X, uv) in enumerate(zip(W3d, Q2d), start=1):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(i); v_p.set_estimate(X.astype(float)); v_p.set_fixed(True)
        optimizer.add_vertex(v_p)

        e = g2o.EdgeProjectXYZ2UV()
        e.set_vertex(0, v_p); e.set_vertex(1, v_se3)
        e.set_measurement(uv.astype(float))
        e.set_information(np.eye(2))
        e.set_parameter_id(0, 0)
        e.set_robust_kernel(huber)
        optimizer.add_edge(e)

    optimizer.initialize_optimization()
    optimizer.optimize(iters)

    est = v_se3.estimate()
    R = est.rotation().toRotationMatrix(); t = est.translation()
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    return T

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser("Pose-only (g2o) from AnyLoc matches with ORB-style init")
    ap.add_argument("--root", required=True)        # outputs/00
    ap.add_argument("--io-dir", required=True)      # outputs/00/anyloc_io/sm_00001
    ap.add_argument("--matches", required=True)     # .../matches.csv
    ap.add_argument("--K", required=True, help="9 comma-separated (row-major)")
    ap.add_argument("--dist", default="0,0,0,0,0")
    ap.add_argument("--max-pairs", type=int, default=10000)
    ap.add_argument("--save-pose", default=None)
    args = ap.parse_args()

    K = np.fromstring(args.K, sep=",", dtype=np.float64).reshape(3,3)
    dist = np.fromstring(args.dist, sep=",", dtype=np.float64).reshape(-1,1)

    Q2d, W3d, AB, meta = gather_correspondences(args.root, args.io_dir, args.matches, args.max_pairs)
    print(f"[pose] gathered {len(Q2d)} 2D-3D pairs")
    if len(Q2d) < 6:
        raise SystemExit("Not enough correspondences after filtering (need ≥6).")

    T0, inl = None, None

    # 1) AP3P
    res = try_pnp_ransac(W3d, Q2d, K, dist, cv2.SOLVEPNP_AP3P, thr_px=8.0, iters=4000)
    if res is None:
        print("[init] AP3P failed; trying EPNP…")
        res = try_pnp_ransac(W3d, Q2d, K, dist, cv2.SOLVEPNP_EPNP, thr_px=12.0, iters=6000)

    # 2) IPPE (planar PnP)
    if res is None:
        print("[init] EPNP failed; trying IPPE…")
        res = try_ippe(Q2d, W3d, K, dist, meta)

    # 3) Homography decomposition (ORB-style planar init)
    if res is None:
        print("[init] IPPE failed; trying Homography decomposition…")
        res = try_homography(Q2d, AB, K)

    if res is None:
        raise SystemExit("All initializers failed (AP3P/EPNP/IPPE/H). Check matches and intrinsics.")

    T0, inl = res
    print(f"[init] OK with {len(inl)} inliers → BA")

    # g2o BA with inliers only
    T = g2o_refine_pose(T0, Q2d[inl], W3d[inl], K, iters=30)
    print("[pose] SE3 (world←camera):\n", np.array2string(T, formatter={'float_kind':lambda x: f'{x: .6f}'}))

    if args.save_pose:
        os.makedirs(os.path.dirname(args.save_pose), exist_ok=True)
        np.save(args.save_pose, T.astype(np.float64))
        print(f"[pose] saved → {args.save_pose}")

if __name__ == "__main__":
    main()
