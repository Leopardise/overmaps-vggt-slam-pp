import numpy as np
import pycolmap
import os
from pathlib import Path
from repo_paths import POSES, sparse_dir

SPARSE_BASE = None  # resolved per-scene via sparse_dir()
OUT_BASE    = str(POSES)

SCENES = [
    ("acc84284-a9c7-4c6d-b8d7-429d1878b36d", "Waterfront_Beach_Morning_Sunny",    "Waterfront;Beach",          "Morning, Sunny",    149.50),
    ("cf193a84-cb9a-4416-8e35-c5bb726d18ab", "Urban_CityPark_Afternoon_Rainy",    "Urban;City Park",           "Afternoon, Rainy",   50.63),
    ("077440b5-e71f-4a08-840a-d8b5737d111f", "Natural_Forest_Morning_Sunny",      "Natural;Forest",            "Morning, Sunny",     74.04),
    ("8e2217c3-bb88-45ad-ba3f-7c9d7b3232f5", "Urban_CityStreet_Afternoon_Clear",  "Urban;City Street",         "Afternoon, Clear",  118.49),
    ("dd329eef-e796-4915-8ce0-dc8e3e844840", "Urban_CityStreet_Night_Clear",      "Urban;City Street (Night)", "Night, Clear",       62.53),
    ("6dcff26a-ed5d-46c0-b78c-c8af7124dc03", "Interior_Mall_Morning_Indoor",      "Interior;Shopping Mall",    "Morning, Indoor",    51.93),
]

def load_colmap_gt(uuid, folder):
    r = pycolmap.Reconstruction(str(sparse_dir(uuid, folder)))
    imgs = sorted(r.images.values(), key=lambda x: x.name)
    return [img.projection_center() for img in imgs]

def load_poses(path):
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
                poses.append(np.array(vals[1:4]))
    return poses

def umeyama(src, dst):
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc**2).sum() / n
    H = sc.T @ dc / n
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    scale = (S * np.array([1,1,d])).sum() / (var_s + 1e-10)
    t = mu_d - scale * R @ mu_s
    return (scale * (R @ src.T)).T + t

def ate(est, gt):
    n = min(len(est), len(gt))
    E, G = np.array(est[:n]), np.array(gt[:n])
    E_al = umeyama(E, G)
    errs = np.linalg.norm(E_al - G, axis=1)
    return errs, n

print("\n" + "="*100)
print("ATE EVALUATION — ALL 6 SCENES — VGGT-SLAM++ vs COLMAP Ground Truth")
print("="*100)
print(f"{'Scene':<30} {'Condition':<20} {'TrajLen':>8} {'Frames':>7} {'Vanilla':>9} {'Optimised':>10} {'Rel%':>6} {'LC Effect'}")
print("-"*100)

results = []
for uuid, folder, label, cond, traj_len in SCENES:
    root = os.path.join(OUT_BASE, folder)
    try:
        gt = load_colmap_gt(uuid, folder)
    except Exception as e:
        print(f"{label:<30} GT load failed: {e}")
        continue

    vanilla   = load_poses(os.path.join(root, "poses_vanilla.txt"))
    optimised = load_poses(os.path.join(root, "poses.txt"))

    e_v, n_v = ate(vanilla, gt)
    e_o, n_o = ate(optimised, gt)

    rel_pct = (e_o.mean() / traj_len) * 100
    impr = (e_v.mean() - e_o.mean()) / e_v.mean() * 100
    effect = f"{impr:+.1f}%"

    print(f"{label:<30} {cond:<20} {traj_len:>7.1f}m {n_o:>7} {e_v.mean():>8.3f}m {e_o.mean():>9.3f}m {rel_pct:>5.1f}% {effect}")
    results.append((label, cond, traj_len, n_o, e_v.mean(), e_o.mean(), rel_pct, impr,
                    e_o.max(), np.sqrt((e_o**2).mean())))

print("="*100)
print(f"\n{'SUMMARY':}")
print(f"Mean vanilla ATE    : {np.mean([r[4] for r in results]):.3f} m")
print(f"Mean optimised ATE  : {np.mean([r[5] for r in results]):.3f} m")
print(f"Mean relative error : {np.mean([r[6] for r in results]):.1f}%")
print(f"Mean LC improvement : {np.mean([r[7] for r in results]):+.1f}%")

# Save for report
import json
with open(Path(__file__).with_name("ate_all6_results.json"), "w") as f:
    json.dump([{"scene": r[0], "condition": r[1], "traj_len": r[2], "frames": r[3],
                "vanilla_ate": round(r[4],4), "optimised_ate": round(r[5],4),
                "relative_pct": round(r[6],2), "lc_improvement": round(r[7],2),
                "max_error": round(r[8],4), "rmse": round(r[9],4)} for r in results], f, indent=2)
print("\nSaved to ate_all6_results.json")
