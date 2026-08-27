# Data in this repo

OverMaps-1K is **~3.9 TB / 1,000 scenes**. GitHub will not host that. A single zip of even the six evaluated scenes is **~3.7 GB**, which also exceeds GitHub’s **100 MB per-file** limit. Scene-level zips (400–800 MB each) hit the same wall.

So the sequences VGGT-SLAM++ actually ran are stored as **individual JPEGs** (largest file ~4 MB) plus COLMAP `sparse/`:

```
OverMaps-eval6/<SceneName>__<uuid8>/
  images/     undistorted RGB (PINHOLE)
  sparse/0/   cameras.bin  images.bin  points3D.bin
```

| Folder | Scene | Frames |
|---|---|---:|
| `Waterfront_Beach_Morning_Sunny__acc84284` | Waterfront;Beach | 299 |
| `Urban_CityPark_Afternoon_Rainy__cf193a84` | Urban;City Park (rain) | 299 |
| `Natural_Forest_Morning_Sunny__077440b5` | Natural;Forest | 299 |
| `Urban_CityStreet_Afternoon_Clear__8e2217c3` | Urban;City Street | 297 |
| `Urban_CityStreet_Night_Clear__dd329eef` | City Street (night) | 300 |
| `Interior_Mall_Morning_Indoor__6dcff26a` | Interior;Mall | 300 |

`dataset_manifest.csv` is the 1,000-scene index from the company release (ids + asset paths, no RGB).

`insta360/` holds the three X5 clips as **already-extracted 90° forward crops** plus ARKit `poses_*.csv`. Raw `.insv` files are not included.

Full 1K set: Hugging Face [`OverTheReality/OverMaps_1k`](https://huggingface.co/datasets/OverTheReality/OverMaps_1k). © Over The Reality; shared with us for SLAM research. People and plates are inpainted at source.

To point the audit scripts at a local full dump:

```bash
export OVERMAPS_ROOT=/path/to/OverMaps-1K
```
