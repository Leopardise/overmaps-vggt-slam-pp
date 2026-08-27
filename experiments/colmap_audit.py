import os
import numpy as np
import json

# install pycolmap if needed
try:
    import pycolmap
except:
    os.system("pip install pycolmap --break-system-packages")
    import pycolmap

from pathlib import Path
from repo_paths import overmaps_1k, images_dir, sparse_dir

_root = overmaps_1k()
SPARSE_DIR = str(_root / "sparse") if _root else None
IMAGES_DIR = str(_root / "images") if _root else None

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

print(f"\n{'UUID':<10} {'Scene':<35} {'Time':<12} {'Wx':<15} {'Imgs_reg':<10} {'Points3D':<10} {'ReprErr':<10} {'CamModel':<20} {'ImgCount'}")
print("-"*130)

for uid, (stype, tod, wx) in scene_meta.items():
    sparse_path = os.path.join(SPARSE_DIR, uid, "0")
    images_path = os.path.join(IMAGES_DIR, uid)

    # count actual images on disk
    img_count = len(os.listdir(images_path)) if os.path.exists(images_path) else -1

    if not os.path.exists(sparse_path):
        print(f"{uid[:8]:<10} {'SPARSE MISSING'}")
        continue

    try:
        r = pycolmap.Reconstruction(sparse_path)
        errors = [p.error for p in r.points3D.values()]
        mean_err = np.mean(errors) if errors else -1
        cam_model = list(r.cameras.values())[0].model_name if r.cameras else "?"

        row = {
            "uid": uid,
            "scene_type": stype,
            "time": tod,
            "weather": wx,
            "registered_images": len(r.images),
            "points3D": len(r.points3D),
            "mean_reproj_error": round(mean_err, 4),
            "camera_model": cam_model,
            "images_on_disk": img_count,
            "registration_ratio": round(len(r.images) / img_count, 3) if img_count > 0 else -1
        }
        results.append(row)

        print(f"{uid[:8]:<10} {stype:<35} {tod:<12} {wx:<15} {len(r.images):<10} {len(r.points3D):<10} {mean_err:<10.3f} {cam_model:<20} {img_count}")

    except Exception as e:
        print(f"{uid[:8]:<10} [ERROR] {e}")

# Save results
out_path = str(Path(__file__).with_name("colmap_results.json"))
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")

# Summary stats
if results:
    errs = [r['mean_reproj_error'] for r in results if r['mean_reproj_error'] > 0]
    regs = [r['registration_ratio'] for r in results if r['registration_ratio'] > 0]
    print(f"\n========== SUMMARY ==========")
    print(f"Scenes analyzed       : {len(results)}")
    print(f"Mean reproj error     : {np.mean(errs):.3f} px  (std: {np.std(errs):.3f})")
    print(f"Mean registration     : {np.mean(regs):.3f}      (std: {np.std(regs):.3f})")
    print(f"Best reproj error     : {min(errs):.3f} px")
    print(f"Worst reproj error    : {max(errs):.3f} px")
    cam_models = [r['camera_model'] for r in results]
    print(f"Camera models seen    : {set(cam_models)}")
