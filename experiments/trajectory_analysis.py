import os, numpy as np, json, pycolmap

from pathlib import Path
from repo_paths import overmaps_1k, sparse_dir

_root = overmaps_1k()
SPARSE_DIR = str(_root / "sparse") if _root else None

scene_meta = {
    "acc84284-a9c7-4c6d-b8d7-429d1878b36d": ("Waterfront;Beach",        "Morning",   "Sunny"),
    "99bc9580-1fce-4024-b474-1dc0b32699d6": ("Urban;City Square",        "Morning",   "Clear"),
    "cfc399dc-702b-41dc-b94c-53d54998aa70": ("Natural Landscape;Park",   "Afternoon", "Sunny"),
    "904fc0bf-f026-493c-86ca-a78bfd613810": ("Urban;City Park",          "Morning",   "Sunny"),
    "cf193a84-cb9a-4416-8e35-c5bb726d18ab": ("Urban;City Park",          "Afternoon", "Rainy"),
    "077440b5-e71f-4a08-840a-d8b5737d111f": ("Natural Landscape;Forest", "Morning",   "Sunny"),
    "ee1cffaf-99ab-4035-bd41-9cdcc24d8de3": ("Urban;Modern Courtyard",   "Night",     "Clear"),
    "8e1e4107-6d3d-4e64-b8b1-cb9eedbaf553": ("Natural Landscape;Park",   "Afternoon", "Sunny"),
    "6dcff26a-ed5d-46c0-b78c-c8af7124dc03": ("Interior;Shopping Mall",   "Morning",   "Indoor"),
    "6562f66d-a92f-4c7d-87c1-c86fb1d240e0": ("Urban;City Square",        "Afternoon", "Partly Cloudy"),
    "dd329eef-e796-4915-8ce0-dc8e3e844840": ("Urban;City Street",        "Night",     "Clear"),
    "8e2217c3-bb88-45ad-ba3f-7c9d7b3232f5": ("Urban;City Street",        "Afternoon", "Clear"),
    "1c4784a9-8e27-4422-b250-cbdf0409315c": ("Urban;City Street",        "Afternoon", "Sunny"),
    "0983df21-c5ac-4451-bd3b-e09cdaaa9897": ("Natural Landscape;Forest", "Afternoon", "Sunny"),
    "72217f1d-a678-4154-98eb-29be4547b236": ("Waterfront;Beach",         "Dusk",      "Partly Cloudy"),
    "d176504f-2fdc-4c05-9030-b02216ca1b08": ("Interior;Arcade",          "Morning",   "Indoor"),
}

results = []
print(f"\n{'UUID':<10} {'Scene':<35} {'Time':<12} {'TrajLen(m)':<12} {'SceneRadius(m)':<16} {'AvgBaseline(cm)'}")
print("-"*110)

for uid, (stype, tod, wx) in scene_meta.items():
    sparse_path = os.path.join(SPARSE_DIR, uid, "0")
    try:
        r = pycolmap.Reconstruction(sparse_path)

        # projection_center() is the correct method in pycolmap v4
        sorted_imgs = sorted(r.images.values(), key=lambda x: x.name)
        centers = np.array([img.projection_center() for img in sorted_imgs])

        diffs = np.diff(centers, axis=0)
        traj_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
        centroid = centers.mean(axis=0)
        scene_radius = float(np.max(np.linalg.norm(centers - centroid, axis=1)))
        avg_baseline = float(np.mean(np.linalg.norm(diffs, axis=1))) * 100

        row = {
            "uid": uid, "scene_type": stype, "time": tod, "weather": wx,
            "trajectory_length_m": round(traj_len, 2),
            "scene_radius_m": round(scene_radius, 2),
            "avg_baseline_cm": round(avg_baseline, 2),
            "num_frames": len(r.images)
        }
        results.append(row)
        print(f"{uid[:8]:<10} {stype:<35} {tod:<12} {traj_len:<12.2f} {scene_radius:<16.2f} {avg_baseline:.2f}")

    except Exception as e:
        print(f"{uid[:8]:<10} [ERROR] {e}")

with open(Path(__file__).with_name("trajectory_results.json"), "w") as f:
    json.dump(results, f, indent=2)

if results:
    print(f"\n========== SUMMARY ==========")
    print(f"Mean trajectory length : {np.mean([r['trajectory_length_m'] for r in results]):.2f} m")
    print(f"Mean scene radius      : {np.mean([r['scene_radius_m'] for r in results]):.2f} m")
    print(f"Mean baseline          : {np.mean([r['avg_baseline_cm'] for r in results]):.2f} cm")
print("Done. Saved to trajectory_results.json")
