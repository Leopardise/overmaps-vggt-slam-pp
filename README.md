# OverMaps-1K × VGGT-SLAM++

Technical evaluation of the **OverMaps-1K** dataset (Over The Reality, Udine) and three **Insta360 X5** clips, run with [VGGT-SLAM++](https://arxiv.org/abs/2604.06830) (Mandal, Kumar, Harithas, Arora; CVPRW 2026).

This repo is the public pack: VGGT-SLAM++ eval scripts, ATE tables, pose logs, the six OverMaps scenes we actually ran, and the three Insta360 samples. The original working tree and the full 3.9 TB 1K dump stay on the lab disk.

The full OverMaps-1K release is **~3.9 TB / 1,000 scenes**. GitHub cannot host that. We ship **only the six sequences used for VGGT-SLAM++ ATE** plus COLMAP `sparse/` and the 1K `dataset_manifest.csv`. Dense SLAM maps (`vggt_outputs/`, ~270 GB) stay on the scratch disk.

**New here?** [`quickrun.txt`](quickrun.txt) · full write-up: [`REPORT.md`](REPORT.md)

## Headline numbers

| Criterion | Finding |
|---|---|
| COLMAP registration (16-scene audit) | 100% frames, PINHOLE, mean reproj. 1.228 px |
| Image sharpness | mean Laplacian var. 705.8 |
| VGGT-SLAM++ relative ATE (6 OverMaps scenes) | **4.2–5.9%, mean 4.7%** |
| Loop-closure ΔATE | ~−0.5% (neutral) |
| Insta360 90° crops vs ARKit GT | relative ATE **14.8–17.7%, mean 16.1%** |

OverMaps-1K ATE after Umeyama Sim(3) vs COLMAP (`results/ate` via `experiments/ate_all6_results.json`):

| Scene | Traj. | Vanilla ATE | Optimised ATE | Rel. | LC |
|---|---:|---:|---:|---:|---:|
| Waterfront;Beach | 149.5 m | 6.416 | 6.337 | 4.2% | +1.2% |
| Urban;City Park (rain) | 50.6 m | 2.238 | 2.287 | 4.5% | −2.2% |
| Natural;Forest | 74.0 m | 3.103 | 3.116 | 4.2% | −0.4% |
| Urban;City Street | 118.5 m | 7.005 | 7.040 | 5.9% | −0.5% |
| City Street (night) | 62.5 m | 2.661 | 2.686 | 4.3% | −0.9% |
| Interior;Mall | 51.9 m | 2.677 | 2.676 | 5.2% | +0.0% |

## Layout

```
slam/                      VGGT-SLAM++ (method code; no off-the-shelf VGGT-1B)
experiments/               audit, COLMAP, sharpness, trajectory, ATE, runners
results/poses/             vanilla + optimised TUM pose logs
results/anyloc/            loop_pairs / loop_votes CSVs
results/logs/              runner logs
data/dataset_manifest.csv  1,000-scene index from the company release
data/OverMaps-eval6/       6 eval scenes: images/ + sparse/ (COLMAP)
data/insta360/             3 clips: perspective crops + ARKit poses_*.csv
REPORT.md                  evaluation report (June 2026)
```

## Reproduce ATE (no GPU)

```bash
python experiments/compute_ate_all6.py   # edit SPARSE_BASE / OUT_BASE to this tree
```

Pose logs are already in `results/poses/`.

## Re-run VGGT-SLAM++ (GPU)

```bash
cd slam && ./setup.sh          # downloads VGGT-1B via Hugging Face, not vendored
python main.py --image_folder ../data/OverMaps-eval6/Waterfront_Beach_Morning_Sunny__acc84284/images \
  --log_results --log_path ../results/poses/Waterfront_Beach_Morning_Sunny/poses.txt
```

Insta360 clips were converted from equirectangular to a **1024×1024, 90° forward crop** (`experiments/insta360_pipeline.py`) before the same pipeline.

## Dataset credit

OverMaps-1K © Over The Reality. Shared with us for SLAM research. Full 1K set: Hugging Face `OverTheReality/OverMaps_1k`. People/plates are inpainted at source.

## Citation

```
@article{mandal2026vggt-slam-pp,
  title={VGGT-SLAM++},
  author={Mandal, Avilasha and Kumar, Rajesh and Harithas, Sudarshan Sunil and Arora, Chetan},
  journal={arXiv preprint arXiv:2604.06830},
  year={2026}
}
```
