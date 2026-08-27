import numpy as np
import pycolmap
import os

from repo_paths import POSES, sparse_dir

SPARSE_BASE = None  # resolved per-scene via sparse_dir()
OUT_BASE    = str(POSES)

SCENES = [
    ("acc84284-a9c7-4c6d-b8d7-429d1878b36d", "Waterfront_Beach_Morning_Sunny",  "Waterfront;Beach"),
    ("cf193a84-cb9a-4416-8e35-c5bb726d18ab", "Urban_CityPark_Afternoon_Rainy",  "Urban;City Park (Rainy)"),
    ("077440b5-e71f-4a08-840a-d8b5737d111f", "Natural_Forest_Morning_Sunny",    "Natural;Forest"),
]

def load_colmap_gt_ordered(uuid, folder):
    """Load GT as ordered list matching sorted image names (same order as main.py loads images)."""
    r = pycolmap.Reconstruction(str(sparse_dir(uuid, folder)))
    # sort images by name — same as main.py's sorted glob
    imgs = sorted(r.images.values(), key=lambda x: x.name)
    centers = [img.projection_center() for img in imgs]
    names   = [img.name for img in imgs]
    return centers, names

def load_poses_txt_ordered(path):
    """Load poses as ordered list: row i = frame i. Columns: x y z (translation)."""
    if not os.path.isfile(path):
        return []
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = list(map(float, line.split()))
            if len(vals) >= 4:
                poses.append(np.array(vals[1:4]))  # x y z
    return poses

def umeyama_alignment(src, dst):
    """
    Similarity transform alignment (scale + rotation + translation).
    src, dst: (N, 3) numpy arrays.
    Returns aligned src.
    """
    n = src.shape[0]
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = (src_c**2).sum() / n
    H = src_c.T @ dst_c / n
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    scale = (S * np.array([1,1,d])).sum() / (var_s + 1e-10)
    t = mu_d - scale * R @ mu_s
    aligned = (scale * (R @ src.T)).T + t
    return aligned

def compute_ate(est_poses, gt_centers, label):
    n = min(len(est_poses), len(gt_centers))
    if n < 3:
        print(f"  {label}: insufficient frames ({n})")
        return None
    E = np.array(est_poses[:n])
    G = np.array(gt_centers[:n])
    E_aligned = umeyama_alignment(E, G)
    errs = np.linalg.norm(E_aligned - G, axis=1)
    print(f"  {label}:")
    print(f"    Frames  : {n}")
    print(f"    ATE mean  : {errs.mean():.4f} m")
    print(f"    ATE median: {np.median(errs):.4f} m")
    print(f"    ATE RMSE  : {np.sqrt((errs**2).mean()):.4f} m")
    print(f"    ATE max   : {errs.max():.4f} m")
    return errs

print("\n" + "="*60)
print("ATE EVALUATION — VGGT-SLAM++ vs COLMAP Ground Truth")
print("="*60)

results = []
for uuid, folder, label in SCENES:
    root = os.path.join(OUT_BASE, folder)

    print(f"\n{'─'*60}")
    print(f"SCENE: {label}")
    print(f"{'─'*60}")

    try:
        gt_centers, gt_names = load_colmap_gt_ordered(uuid, folder)
        print(f"  GT frames: {len(gt_centers)}")
    except Exception as e:
        print(f"  GT load failed: {e}")
        continue

    vanilla   = load_poses_txt_ordered(os.path.join(root, "poses_vanilla.txt"))
    optimised = load_poses_txt_ordered(os.path.join(root, "poses.txt"))

    print(f"\n  [Vanilla — no loop closure] ({len(vanilla)} poses)")
    e_v = compute_ate(vanilla, gt_centers, "Vanilla")

    print(f"\n  [Optimised — with loop closure] ({len(optimised)} poses)")
    e_o = compute_ate(optimised, gt_centers, "Optimised")

    if e_v is not None and e_o is not None:
        improvement = (e_v.mean() - e_o.mean()) / e_v.mean() * 100
        print(f"\n  Loop closure improvement: {improvement:+.1f}%")
        results.append((label, e_v.mean(), e_o.mean(), improvement,
                        len(vanilla), e_v.max(), e_o.max()))

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Scene':<30} {'Vanilla ATE':<14} {'Optimised ATE':<15} {'Improvement'}")
print("-"*75)
for label, v_mean, o_mean, impr, frames, v_max, o_max in results:
    print(f"{label:<30} {v_mean:.4f} m       {o_mean:.4f} m        {impr:+.1f}%")
print("="*60)
