import os, cv2, numpy as np, json

from pathlib import Path
from repo_paths import overmaps_1k, images_dir

_root = overmaps_1k()
IMAGES_DIR = str(_root / "images") if _root else None

scene_meta = {
    "acc84284-a9c7-4c6d-b8d7-429d1878b36d": ("Waterfront;Beach",        "Morning",   "Sunny"),
    "99bc9580-1fce-4024-b474-1dc0b32699d6": ("Urban;City Square",        "Morning",   "Clear"),
    "cfc399dc-702b-41dc-b94c-53d54998aa70": ("Natural Landscape;Park",   "Afternoon", "Sunny"),
    "904fc0bf-f026-493c-86ca-a78bfd613810": ("Urban;City Park",          "Morning",   "Sunny"),
    "cf193a84-cb9a-4416-8e35-c5bb726d18ab": ("Urban;City Park",          "Afternoon", "Rainy"),
    "077440b5-e71f-4a08-840a-d8b5737d111f": ("Natural Landscape;Forest", "Morning",   "Sunny"),
    "8e1e4107-6d3d-4e64-b8b1-cb9eedbaf553": ("Natural Landscape;Park",   "Afternoon", "Sunny"),
    "6dcff26a-ed5d-46c0-b78c-c8af7124dc03": ("Interior;Shopping Mall",   "Morning",   "Indoor"),
    "6562f66d-a92f-4c7d-87c1-c86fb1d240e0": ("Urban;City Square",        "Afternoon", "Partly Cloudy"),
    "8e2217c3-bb88-45ad-ba3f-7c9d7b3232f5": ("Urban;City Street",        "Afternoon", "Clear"),
    "1c4784a9-8e27-4422-b250-cbdf0409315c": ("Urban;City Street",        "Afternoon", "Sunny"),
    "0983df21-c5ac-4451-bd3b-e09cdaaa9897": ("Natural Landscape;Forest", "Afternoon", "Sunny"),
    "72217f1d-a678-4154-98eb-29be4547b236": ("Waterfront;Beach",         "Dusk",      "Partly Cloudy"),
}

def analyze_scene(uid, stype, tod, wx):
    img_dir = os.path.join(IMAGES_DIR, uid)
    if not os.path.exists(img_dir):
        return None

    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
    if not files:
        return None

    # Sample every 10th frame for speed
    sample = files[::10]

    blurs, brightnesses, contrasts = [], [], []
    resolutions = set()

    for fname in sample:
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur: Laplacian variance (higher = sharper)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blurs.append(blur)

        # Brightness: mean pixel value
        brightnesses.append(np.mean(gray))

        # Contrast: std of pixel values
        contrasts.append(np.std(gray))

        # Resolution
        resolutions.add(f"{img.shape[1]}x{img.shape[0]}")

    return {
        "uid": uid,
        "scene_type": stype,
        "time": tod,
        "weather": wx,
        "total_frames": len(files),
        "sampled": len(sample),
        "mean_sharpness": round(float(np.mean(blurs)), 1),
        "min_sharpness": round(float(np.min(blurs)), 1),
        "mean_brightness": round(float(np.mean(brightnesses)), 1),
        "mean_contrast": round(float(np.mean(contrasts)), 1),
        "resolutions": list(resolutions),
    }

results = []
print(f"\n{'UUID':<10} {'Scene':<35} {'Time':<12} {'Wx':<15} {'Frames':<8} {'Sharpness':<12} {'Brightness':<12} {'Contrast':<10} {'Resolution'}")
print("-"*130)

for uid, (stype, tod, wx) in scene_meta.items():
    r = analyze_scene(uid, stype, tod, wx)
    if r:
        results.append(r)
        print(f"{uid[:8]:<10} {stype:<35} {tod:<12} {wx:<15} {r['total_frames']:<8} {r['mean_sharpness']:<12.1f} {r['mean_brightness']:<12.1f} {r['mean_contrast']:<10.1f} {r['resolutions']}")
    else:
        print(f"{uid[:8]:<10} [MISSING]")

with open(Path(__file__).with_name("image_quality_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n========== SUMMARY ==========")
if results:
    print(f"Mean sharpness  : {np.mean([r['mean_sharpness'] for r in results]):.1f}  (higher=sharper, >100 is good)")
    print(f"Mean brightness : {np.mean([r['mean_brightness'] for r in results]):.1f}  (0-255 scale, ~128 is neutral)")
    print(f"Mean contrast   : {np.mean([r['mean_contrast'] for r in results]):.1f}")
    all_res = set(r for row in results for r in row['resolutions'])
    print(f"Resolutions seen: {all_res}")
print("Done. Results saved to image_quality_results.json")
