# OverMaps-1K Dataset — Technical Evaluation Report

**From the perspective of visual SLAM and 3D reconstruction research**

Avilasha Mandal  
First Author, VGGT-SLAM++  
Department of Computer Science and Engineering  
Indian Institute of Technology Delhi  
June 2026

---

## 1. Executive summary

This report evaluates **OverMaps-1K**, a 1,000-scene, 3.9 TB real-world mapping dataset from Over The Reality (Udine, Italy), for compatibility with **VGGT-SLAM++**.

A stratified sample of **16 scenes** was audited (COLMAP quality, sharpness, scale, 1K metadata). **VGGT-SLAM++** was then run on **six** of those scenes with ATE vs COLMAP. Three **Insta360 X5** 360° clips were reviewed as a supplement.

| Criterion | Finding | Assessment |
|---|---|---|
| COLMAP format | PINHOLE, standard binary | Fully compatible |
| Frame registration | 100% on all 16 scenes | Excellent |
| Mean reprojection error | 1.228 px (0.868–1.468) | Good, not sub-pixel |
| Image sharpness | Mean 705.8 | Excellent in good light |
| LiDAR depth | 481/1000 scenes (~48%) | As claimed |
| Scene diversity | Urban-heavy, daytime-dominant | Modest |
| Adverse conditions | 3.8% night, 2.9% rain, 0.2% snow | Limited |
| Portrait 9:16 | Center-crop to 518×518; ~44% vertical FoV lost | Worth clarifying |
| VGGT-SLAM++ ATE (6 scenes) | ~4.2–5.9% relative, mean **4.7%** | Consistent |
| Loop closure | Marginal on this sample | Scene-dependent |

## 2. Dataset overview

OverMaps-1K is a research subset of ~155,000 scenes, captured on consumer iOS/Android via an AR-guided hexagonal tiling protocol (ARKit/ARCore VIO, metric scale from frame 1). Triggers: 10 cm translation or 5° rotation. Post-process: privacy inpainting (YOLOv6 + LaMa), pose refinement (NetVLAD + HLOC + ALIKED + LightGlue + COLMAP + pixSfM), Qwen3-VL labels.

### Per-scene folders

| Folder | Contents | SLAM use |
|---|---|---|
| `images/` | Undistorted JPEG (~400+ / scene) | Primary input |
| `sparse/` | COLMAP `cameras.bin`, `images.bin`, `points3D.bin` | GT poses |
| `depth_est/` | ICG-MVS `.pfm` | Estimated, not metric GT |
| `depths/` | iPhone LiDAR (~48%) | Metric depth |
| `3dgs/` | Gaussian splat `.ply` | Visual QA |
| `images_raw/` | Pre-undistortion | Distortion research |
| `masks_images/` | Privacy masks | People inpainted |
| `images-csv/` | Timestamp, GPS, pose | Frame metadata |
| `yolo11_labels/` | Per-frame detections | Content labels |

### 1,000-scene composition (`data/dataset_manifest.csv`)

- **Type:** Urban;City Street 316, City Square 112; 43% street/square; 224 types, long-tailed.
- **Time:** Afternoon 480, Morning 459, Night 38 (3.8%).
- **Weather:** Sunny 300, Overcast 279; rain 2.9%, snow 0.2%.
- **Crowd:** 78% low or empty.
- **LiDAR:** 481 scenes.

## 3. Evaluation sample (16 scenes)

| UUID (short) | Type | Time | Weather |
|---|---|---|---|
| acc84284 | Waterfront;Beach | Morning | Sunny |
| 99bc9580 | Urban;City Square | Morning | Clear |
| cfc399dc | Natural Landscape;Park | Afternoon | Sunny |
| 904fc0bf | Urban;City Park | Morning | Sunny |
| cf193a84 | Urban;City Park | Afternoon | Rainy |
| 077440b5 | Natural Landscape;Forest | Morning | Sunny |
| ee1cffaf | Urban;Modern Courtyard | Night | Clear |
| 8e1e4107 | Natural Landscape;Park | Afternoon | Sunny |
| 6dcff26a | Interior;Shopping Mall | Morning | Indoor |
| 6562f66d | Urban;City Square | Afternoon | Partly Cloudy |
| dd329eef | Urban;City Street | Night | Clear |
| 8e2217c3 | Urban;City Street | Afternoon | Clear |
| 1c4784a9 | Urban;City Street | Afternoon | Sunny |
| 0983df21 | Natural Landscape;Forest | Afternoon | Sunny |
| 72217f1d | Waterfront;Beach | Dusk | Partly Cloudy |
| d176504f | Interior;Arcade | Morning | Indoor |

**VGGT-SLAM++ ATE scenes (in `data/OverMaps-eval6/`):** acc84284, cf193a84, 077440b5, 8e2217c3, dd329eef, 6dcff26a.

## 4. COLMAP quality

All 16 scenes: 100% registration, PINHOLE. Mean reprojection **1.228 px** (std 0.191). Rainy / indoor scenes have fewer 3D points (mall 61k vs ~150k typical). Documentation claims sub-pixel after pixSfM — we ask providers for the exact refinement settings.

Raw table: `experiments/colmap_results.json`.

## 5. Image quality

Mean sharpness 705.8 (Laplacian variance). Daylight landscapes 1500–2000; night courtyard 130; mall 61; dusk beach 29. All 16 scenes are **portrait ~9:16** (1036×1842 to 1425×1900), 15 distinct resolutions.

## 6. Scale and trajectory

Mean trajectory **96.74 m** (37–196 m). Mean inter-frame baseline **33.75 cm** (protocol states 10 cm — contributors likely walk faster). Indoor radii 3.3–6.3 m.

## 7. Compatibility with VGGT-SLAM++

JPEG + PINHOLE + COLMAP binaries: no undistortion step. ARKit/ARCore gives metric scale. VGGT resizes with a **center crop to 518×518**, dropping ~44% of vertical FoV on every OverMaps frame. LiDAR is only on ~48% of the 1K set. People are inpainted (no dynamic pedestrians).

## 8–10. Strengths, clarifications, conclusion

Strengths: complete registration, sharp daylight frames, metric VIO, rich annotations, privacy pipeline, walking-scale trajectories.

Clarifications for providers: reprojection vs “sub-pixel” claim; daytime/urban bias; landscape or square crops; canonical resolution; whether `images-csv` is raw VIO or post-pixSfM; whether `depth_est` is metric; Insta360 extracted frames / higher FPS / cube faces.

**Conclusion.** OverMaps-1K is a solid, well-structured set. VGGT-SLAM++ shows a **stable ~4.7% relative ATE** across six very different scenes. We recommend it as a SLAM evaluation set, pending the orientation and 360° pipeline questions.

## 11. Pipeline notes

Portrait crop warning fired on every frame. Vanilla beach run showed scale factors 0.50× → 41.7× across nine submap transitions — expected monocular drift, which is why the full loop-closure back-end was scored in §12.

Environment fixes used at eval time (stay in scratchpad patches if they are host-local): `huggingface-hub==0.36.2`; `.float()` before `.numpy()` on BFloat16; conditional `squeeze`; `--vggt_ckpt` to a cached `model.pt`; wait for GPU memory.

## 12. ATE vs COLMAP (six scenes)

Umeyama similarity (scale + R + t). Mean vanilla ATE 4.017 m; mean optimised 4.024 m; mean relative **4.7%**; mean LC effect **−0.5%**.

Relative error stays in 4.2–5.9% from sunny beach through rain, forest, night street, and mall. LC helped only the beach (+1.2%, 56 loops). Other scenes ±2.3%. Repetitive appearance (beach/forest/rain) weakens VPR; the 43% urban-street 1K set may show stronger LC than this diverse six-scene sample.

JSON: `experiments/ate_all6_results.json`. Poses: `results/poses/`.

## 13. Insta360 X5 (three clips)

Catalog: 706 clips, ~171 h, mostly 8K equirectangular @ 3 FPS; 16% have COLMAP. Geography: Bangkok 43%, then Philippines, South Africa.

Sample folders: extracted perspective crops + `poses_NNN.csv` (phone ARKit) + GPS. No COLMAP on these three. Preprocess: 3840×2160 2:1 → **1024×1024, 90° forward crop**.

| Clip | GT | Traj. | Frames | ATE | Rel. | LC |
|---|---|---:|---:|---:|---:|---:|
| 239eef93 | ARKit poses_39 | 11.2 m | 46 | 1.980 m | 17.7% | 0 |
| 2514321c | ARKit poses_77 | 16.8 m | 94 | 2.493 m | 14.8% | 0 |
| 342137b5 | ARKit poses_489 | 19.1 m | 83 | 3.029 m | 15.9% | 0 |

Mean relative ATE **16.1%** (~3.4× OverMaps). Causes: 75% of the sphere discarded; only two submaps (no LC); ARKit is the **phone**, not the Insta360 (extrinsic/time offset). Cube-face crops or an omnidirectional SLAM stack would be a better 360° benchmark.

Questions for providers: official extracted frames vs SDK; FPS > 3; COLMAP for the other 84%; cube faces; whether `poses_*.csv` can be treated as 360° GT.

---

Avilasha Mandal · VGGT-SLAM++ · IIT Delhi · June 2026
