from __future__ import annotations
import numpy as np

def dem_to_uint8_gray(dem: np.ndarray, clip_lo: float = 1.0, clip_hi: float = 99.0, ignore_zeros: bool = True) -> np.ndarray:
    dem = np.asarray(dem, np.float32)
    if dem.size == 0:
        return np.zeros_like(dem, np.uint8)

    valid = (dem != 0.0) if ignore_zeros else np.isfinite(dem)
    if valid.sum() < 16:
        return np.full(dem.shape, 255, np.uint8)  # too sparse → white

    vals = dem[valid]
    lo = np.percentile(vals, float(clip_lo))
    hi = np.percentile(vals, float(clip_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.full(dem.shape, 255, np.uint8)
        out[valid] = 128
        return out

    x = (dem - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    out = (x * 255.0 + 0.5).astype(np.uint8)
    out[~valid] = 255
    return out
