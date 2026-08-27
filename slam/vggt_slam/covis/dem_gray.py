from __future__ import annotations
import numpy as np
import cv2

def dem_to_u8_gray(
    dem: np.ndarray,
    clip_lo_pct: float = 1.0,
    clip_hi_pct: float = 99.0,
    hole_value: int = 255,
    gauss_sigma_px: float = 1.6,
    gauss_passes: int = 2,
    unsharp_radius_px: float = 1.5,
    unsharp_amount: float = 1.2,
) -> np.ndarray:
    """
    Convert DEM (float32, meters; 0.0 == hole) to uint8 grayscale on white background.
    - Percentile clipping (robust)
    - Multi-pass normalized Gaussian fill *only into holes*
    - Unsharp for crisp edges (optional)
    - Returns HxW uint8, 0..255, holes=255 (white)
    """
    if dem is None or dem.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    dem = dem.astype(np.float32, copy=False)
    H, W = dem.shape
    occ = (dem != 0.0)

    if occ.sum() < 16:
        return np.full((H, W), hole_value, np.uint8)

    # Robust range
    vals = dem[occ]
    lo = np.percentile(vals, float(clip_lo_pct))
    hi = np.percentile(vals, float(clip_hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.full((H, W), hole_value, np.uint8)
        out[occ] = 128
        return out

    # Multi-pass normalized Gaussian fill (into original holes only)
    valid_f = occ.astype(np.float32)
    ksize = int(max(1, round(gauss_sigma_px))) * 2 + 1
    dem_filled = dem.copy()
    for _ in range(max(1, int(gauss_passes))):
        num = cv2.GaussianBlur(dem_filled * valid_f, (ksize, ksize), gauss_sigma_px, borderType=cv2.BORDER_REFLECT)
        den = cv2.GaussianBlur(valid_f,           (ksize, ksize), gauss_sigma_px, borderType=cv2.BORDER_REFLECT) + 1e-9
        filled = num / den
        dem_filled[~occ] = filled[~occ]
        # let newly filled contribute slightly next pass
        valid_f = np.clip(valid_f + (~occ).astype(np.float32)*0.5, 0.0, 1.0)

    # Normalize to 0..255 (occ & filled)
    x = (dem_filled - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    u8 = (x * 255.0 + 0.5).astype(np.uint8)

    # Unsharp for crispness (optional)
    if unsharp_amount > 0 and unsharp_radius_px > 0:
        k = int(max(1, round(unsharp_radius_px))) * 2 + 1
        blur = cv2.GaussianBlur(u8, (k, k), unsharp_radius_px)
        sharp = cv2.addWeighted(u8, 1.0 + unsharp_amount, blur, -unsharp_amount, 0)
        u8 = sharp

    # Holes to white
    u8[~occ] = hole_value
    return u8
