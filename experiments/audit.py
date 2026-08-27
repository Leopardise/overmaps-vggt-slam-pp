import pandas as pd

import os
from pathlib import Path

_PARQUET = os.environ.get(
    "OVERMAPS_PARQUET",
    "/data1/avilasha2/overmaps/OverMaps-1K/data/train-00000-of-00001.parquet",
)
if not Path(_PARQUET).is_file():
    raise SystemExit(
        f"OverMaps parquet not in this repo. Set OVERMAPS_PARQUET to the 1K metadata file.\n"
        f"Looked at: {_PARQUET}"
    )
df = pd.read_parquet(_PARQUET)
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

for col in ['weather', 'time_of_day_algorithmic', 'crowd_density', 'brightness', 'scene_type']:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())

print("\nLiDAR available:", df['depths_path'].notna().sum(), "/ 1000")

# Pick diverse scenes to download
scenes = []
for scene_type in df['scene_type'].value_counts().head(6).index:
    sample = df[df['scene_type'] == scene_type].sample(2, random_state=42)
    scenes.extend(sample['mapping_id'].tolist())

night = df[df['time_of_day_algorithmic'] == 'Night'].sample(2, random_state=42)
scenes.extend(night['mapping_id'].tolist())

indoor = df[df['weather'] == 'Indoor'].sample(2, random_state=42)
scenes.extend(indoor['mapping_id'].tolist())

scenes = list(set(scenes))
print(f"\n--- {len(scenes)} scenes selected for download ---")
for s in scenes:
    row = df[df['mapping_id'] == s].iloc[0]
    print(f"{s}  |  {row['scene_type']}  |  {row['time_of_day_algorithmic']}  |  {row['weather']}")
