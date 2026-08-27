# -*- coding: utf-8 -*-
"""
Streaming cumulative DEM builder with FinderNet-like height colormap and optional edge overlay.

Usage pattern:
    dem = DEMManager(run_root="outputs/run_01", resolution=0.10,
                     snap_every_submaps=10, colormap="turbo",
                     overlay_edges=True, shade=True, mode="cumulative")
    dem.on_submap_finalized(submap)   # call at the end of Solver.add_points()
"""

import os, json, time
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import open3d as o3d
import imageio.v2 as imageio
import cv2
from matplotlib import cm

# ------------------------- utilities -------------------------

def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

def fit_plane_ransac(pts: np.ndarray, dist: float = 0.01, iters: int = 2000):
    """Return plane (a,b,c,d) of ax+by+cz+d=0 via Open3D RANSAC. Requires pts in meters."""
    if pts.shape[0] < 100:
        # Default horizontal plane
        return np.array([0, 0, 1, 0], dtype=np.float64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    model, _ = pcd.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=iters)
    a, b, c, d = model
    # normalize normal
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n) + 1e-12
    return np.array([a / n_norm, b / n_norm, c / n_norm, d / n_norm], dtype=np.float64)

def rotation_to_align_normal_to_z(n: np.ndarray) -> np.ndarray:
    """Compute R (3x3) rotating unit normal n to [0,0,1]."""
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c0 = float(np.dot(n, z))
    if s < 1e-8:
        return np.eye(3, dtype=np.float64)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + (vx @ vx) * ((1 - c0) / (s ** 2))

def np_max_scatter(grid: np.ndarray, gx: np.ndarray, gy: np.ndarray, values: np.ndarray):
    """grid[x,y] = max(grid[x,y], values[i]) with vectorized scatter."""
    np.maximum.at(grid, (gx, gy), values)

# ------------------------- core DEM manager -------------------------

class DEMManager:
    """
    Modes:
      - "cumulative": include ALL submaps so far (default; best for robust edges)
      - "rolling": include last N submaps (set rolling_window_submaps>0)
    """

    def __init__(self,
                 run_root: str,
                 resolution: float = 10,
                 snap_every_submaps: int = 5,
                 colormap: str = "turbo",
                 overlay_edges: bool = True,
                 shade: bool = True,
                 plane_lock_after_submaps: int = 2,
                 rolling_window_submaps: int = 0,    # 0 => disabled; otherwise keep last N submaps
                 mode: str = "cumulative",
                 max_grid: int = 400000,
                 #  4000

                 # cropping/preview options
                 auto_crop: bool = True,
                 crop_margin_px: int = 24,
                 transparent_bg: bool = False,
                    
                 # ↓↓↓ NEW ↓↓↓
                 z_exaggeration: float = 1.5,   # vertical gain (helps 1–7 m pop without clipping 70 m)
                 height_curve: str = "knee",    # "knee" | "gamma" | "log" | "linear"
                 knee_m: float = 7.0,           # knee location in meters (where curve starts compressing)
                 gamma: float = 0.8,            # gamma <1 brightens low heights
                 log_gain: float = 0.3,         # only for 'log' mode; tunes slope
                 vmax_percentile: float = 99.0, # robust upper bound (works for 'linear'/'gamma'/'log')
                 clahe: bool = False,           # local contrast boost (optional)
                 clahe_clip: float = 2.0,
                 clahe_tile: int = 16):
        self.run_root = run_root
        self.resolution = float(resolution)
        self.snap_every = int(snap_every_submaps)
        self.colormap = colormap
        self.overlay_edges = overlay_edges
        self.shade = shade
        self.plane_lock_after = int(plane_lock_after_submaps)
        self.rolling_N = int(rolling_window_submaps)
        self.mode = mode
        self.max_grid = int(max_grid)
        self.z_exaggeration = float(z_exaggeration)
        self.height_curve = str(height_curve).lower()
        self.knee_m = float(knee_m)
        self.gamma = float(gamma)
        self.log_gain = float(log_gain)
        self.vmax_percentile = float(vmax_percentile)
        self.clahe = bool(clahe)
        self.clahe_clip = float(clahe_clip)
        self.clahe_tile = int(clahe_tile)

        self.auto_crop = bool(auto_crop)
        self.crop_margin_px = int(crop_margin_px)
        self.transparent_bg = bool(transparent_bg)

        self.out_dir = os.path.join(self.run_root, "DEMs")
        ensure_dir(self.out_dir)

        # Global state
        self.submap_count = 0
        self.snap_idx = 1

        self.plane_locked = False
        self.plane = np.array([0, 0, 1, 0], dtype=np.float64)   # a,b,c,d normalized
        self.R_align = np.eye(3, dtype=np.float64)              # rotates plane to Z
        self.xmin = None
        self.ymin = None
        self.grid = None   # float32 [nx, ny], maintained as max-height above plane

        # For rolling mode bookkeeping
        self._rolling_points = []  # list of np.ndarray (Ni,3) per submap (only used if rolling mode)

    # ---------- lifecycle ----------

    def on_submap_finalized(self, submap):
        """
        Called once per completed submap from Solver.add_points()
        We pull world points via Submap API, update the DEM, snapshot periodically.
        """
        # 1) fetch world points (confidence-masked, already transformed by H_world_map)
        pts_world = submap.get_points_in_world_frame()  # (N,3), float64
        if pts_world.size == 0:
            return

        # 2) lock global plane once (after a couple of submaps for robustness)
        self.submap_count += 1
        if (not self.plane_locked) and (self.submap_count >= self.plane_lock_after):
            # Fit plane on a moderate random subset to be stable
            sample = pts_world
            if sample.shape[0] > 1_000_000:
                idx = np.random.choice(sample.shape[0], size=1_000_000, replace=False)
                sample = sample[idx]
            plane = fit_plane_ransac(sample, dist=max(0.5 * self.resolution, 0.01), iters=2000)
            n = plane[:3] / (np.linalg.norm(plane[:3]) + 1e-12)
            self.R_align = rotation_to_align_normal_to_z(n)
            self.plane = plane
            self.plane_locked = True

        # 3) choose which points to include (cumulative vs rolling)
        if self.mode == "rolling" and self.rolling_N > 0:
            self._rolling_points.append(pts_world.astype(np.float32))
            if len(self._rolling_points) > self.rolling_N:
                self._rolling_points.pop(0)
            pts_use = np.concatenate(self._rolling_points, axis=0)
        else:
            # cumulative: just use current pts to update grid incrementally (no need to refit past)
            pts_use = pts_world.astype(np.float32)

        # 4) update DEM grid with this submap’s contribution (in place)
        self._update_grid_with_points(pts_use)

        # 5) snapshot periodically
        if self.submap_count % self.snap_every == 0:
            self._write_snapshot()

    # ---------- DEM building ----------

    # def _update_grid_with_points(self, pts_world: np.ndarray):
    #     """Incremental rasterization onto the global DEM grid."""
    #     if pts_world.size == 0:
    #         return

    #     # Center-limit very far outliers (keeps grid size sane)
    #     # First rotate by R_align (stable once plane locked)
    #     pts = (self.R_align @ pts_world.T).T  # (N,3)
    #     # Height above plane: since plane is aligned to Z, we can directly use z after rotating,
    #     # but also keep a fallback plane distance for robustness in lock-warmup.
    #     if self.plane_locked:
    #         heights = np.maximum(pts[:, 2], 0.0)
    #     else:
    #         a, b, c, d = self.plane
    #         nn = max(np.sqrt(a*a + b*b + c*c), 1e-9)
    #         heights = np.maximum((a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / nn, 0.0)

    #     # Compute bounding box in rotated XY
    #     x = pts[:, 0]
    #     y = pts[:, 1]

    #     # Initialize origin and grid on first call
    #     if self.grid is None:
    #         self.xmin = float(np.min(x))
    #         self.ymin = float(np.min(y))
    #         # Start with a modest grid, expand as needed
    #         nx = min(int(np.ceil((np.max(x) - self.xmin) / self.resolution)) + 2, self.max_grid)
    #         ny = min(int(np.ceil((np.max(y) - self.ymin) / self.resolution)) + 2, self.max_grid)
    #         nx = max(nx, 256); ny = max(ny, 256)
    #         self.grid = np.zeros((nx, ny), dtype=np.float32)

    #     # Convert to grid indices
    #     gx = np.floor((x - self.xmin) / self.resolution).astype(np.int32)
    #     gy = np.floor((y - self.ymin) / self.resolution).astype(np.int32)

    #     # # Expand grid if needed (pad with zeros)
    #     # need_nx = int(np.max(gx)) + 1
    #     # need_ny = int(np.max(gy)) + 1
    #     # cur_nx, cur_ny = self.grid.shape
    #     # grow_x = max(0, need_nx - cur_nx)
    #     # grow_y = max(0, need_ny - cur_ny)

    #     # if (grow_x > 0 or grow_y > 0):
    #     #     new_nx = min(cur_nx + max(grow_x, cur_nx // 2, 64), self.max_grid)
    #     #     new_ny = min(cur_ny + max(grow_y, cur_ny // 2, 64), self.max_grid)
    #     #     new_grid = np.zeros((new_nx, new_ny), dtype=np.float32)
    #     #     new_grid[:cur_nx, :cur_ny] = self.grid
    #     #     self.grid = new_grid

    #     # # Clamp to bounds
    #     # gx = np.clip(gx, 0, self.grid.shape[0] - 1)
    #     # gy = np.clip(gy, 0, self.grid.shape[1] - 1)

    #     # # Scatter max heights
    #     # np_max_scatter(self.grid, gx, gy, heights.astype(np.float32))

    #     # --- NEW: grow toward NEGATIVE X/Y if needed (pad on the "front") ---
    #     # pad X (left side)
    #     min_gx = int(gx.min())
    #     if min_gx < 0:
    #         pad_x = -min_gx
    #         cur_nx, cur_ny = self.grid.shape
    #         new_nx = min(cur_nx + pad_x, self.max_grid)
    #         new_grid = np.zeros((new_nx, cur_ny), dtype=np.float32)
    #         new_grid[pad_x:pad_x + cur_nx, :cur_ny] = self.grid
    #         self.grid = new_grid
    #         self.xmin -= pad_x * self.resolution  # shift world origin left
    #         gx += pad_x

    #     # pad Y (top side)
    #     min_gy = int(gy.min())
    #     if min_gy < 0:
    #         pad_y = -min_gy
    #         cur_nx, cur_ny = self.grid.shape
    #         new_ny = min(cur_ny + pad_y, self.max_grid)
    #         new_grid = np.zeros((cur_nx, new_ny), dtype=np.float32)
    #         new_grid[:cur_nx, pad_y:pad_y + cur_ny] = self.grid
    #         self.grid = new_grid
    #         self.ymin -= pad_y * self.resolution  # shift world origin up
    #         gy += pad_y

    #     # --- existing POSITIVE-side growth (keep your code below) ---
    #     need_nx = int(np.max(gx)) + 1
    #     need_ny = int(np.max(gy)) + 1
    #     cur_nx, cur_ny = self.grid.shape
    #     grow_x = max(0, need_nx - cur_nx)
    #     grow_y = max(0, need_ny - cur_ny)

    #     if (grow_x > 0 or grow_y > 0):
    #         new_nx = min(cur_nx + max(grow_x, cur_nx // 2, 64), self.max_grid)
    #         new_ny = min(cur_ny + max(grow_y, cur_ny // 2, 64), self.max_grid)
    #         new_grid = np.zeros((new_nx, new_ny), dtype=np.float32)
    #         new_grid[:cur_nx, :cur_ny] = self.grid
    #         self.grid = new_grid

    #     # Clamp to bounds and scatter
    #     gx = np.clip(gx, 0, self.grid.shape[0] - 1)
    #     gy = np.clip(gy, 0, self.grid.shape[1] - 1)
    #     np_max_scatter(self.grid, gx, gy, heights.astype(np.float32))

    def _update_grid_with_points(self, pts_world: np.ndarray):
        """Incremental rasterization onto the global DEM grid."""
        if pts_world.size == 0:
            return

        # Rotate into plane-aligned coordinates
        pts = (self.R_align @ pts_world.T).T  # (N,3)

        # Height above plane
        if self.plane_locked:
            heights = np.maximum(pts[:, 2], 0.0)
        else:
            a, b, c, d = self.plane
            nn = max(np.sqrt(a*a + b*b + c*c), 1e-9)
            heights = np.maximum((a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / nn, 0.0)

        x = pts[:, 0]
        y = pts[:, 1]

        # Initialize origin and grid on first call
        if self.grid is None:
            self.xmin = float(np.min(x))
            self.ymin = float(np.min(y))
            nx = min(int(np.ceil((np.max(x) - self.xmin) / self.resolution)) + 2, self.max_grid)
            ny = min(int(np.ceil((np.max(y) - self.ymin) / self.resolution)) + 2, self.max_grid)
            nx = max(nx, 256); ny = max(ny, 256)
            self.grid = np.zeros((nx, ny), dtype=np.float32)

        # Convert to grid indices (may be negative if points are left/above current origin)
        gx = np.floor((x - self.xmin) / self.resolution).astype(np.int32)
        gy = np.floor((y - self.ymin) / self.resolution).astype(np.int32)

        # ---- NEW: grow toward NEGATIVE X/Y if needed (pad "front" and shift origin) ----
        min_gx = int(gx.min())
        if min_gx < 0:
            pad_x = -min_gx
            cur_nx, cur_ny = self.grid.shape
            new_nx = min(cur_nx + pad_x, self.max_grid)
            new_grid = np.zeros((new_nx, cur_ny), dtype=np.float32)
            new_grid[pad_x:pad_x + cur_nx, :cur_ny] = self.grid
            self.grid = new_grid
            self.xmin -= pad_x * self.resolution  # shift world origin left
            gx += pad_x

        min_gy = int(gy.min())
        if min_gy < 0:
            pad_y = -min_gy
            cur_nx, cur_ny = self.grid.shape
            new_ny = min(cur_ny + pad_y, self.max_grid)
            new_grid = np.zeros((cur_nx, new_ny), dtype=np.float32)
            new_grid[:cur_nx, pad_y:pad_y + cur_ny] = self.grid
            self.grid = new_grid
            self.ymin -= pad_y * self.resolution  # shift world origin up
            gy += pad_y

        # ---- Existing positive-side growth (unchanged) ----
        need_nx = int(np.max(gx)) + 1
        need_ny = int(np.max(gy)) + 1
        cur_nx, cur_ny = self.grid.shape
        grow_x = max(0, need_nx - cur_nx)
        grow_y = max(0, need_ny - cur_ny)

        if (grow_x > 0 or grow_y > 0):
            new_nx = min(cur_nx + max(grow_x, cur_nx // 2, 64), self.max_grid)
            new_ny = min(cur_ny + max(grow_y, cur_ny // 2, 64), self.max_grid)
            new_grid = np.zeros((new_nx, new_ny), dtype=np.float32)
            new_grid[:cur_nx, :cur_ny] = self.grid
            self.grid = new_grid

        # Clamp to bounds and scatter
        gx = np.clip(gx, 0, self.grid.shape[0] - 1)
        gy = np.clip(gy, 0, self.grid.shape[1] - 1)
        np_max_scatter(self.grid, gx, gy, heights.astype(np.float32))


    def _tone_map_heights(self, H: np.ndarray) -> np.ndarray:
        """
        Map raw heights (meters above plane) into [0,1] with a curve that
        preserves small relief (1–7 m) while not saturating tall structures (e.g., 70 m).
        Returns a float32 image in [0,1].
        """
        if H.size == 0:
            return H.astype(np.float32)

        # Apply global vertical gain first
        Hm = np.maximum(H, 0.0).astype(np.float32) * self.z_exaggeration

        curve = self.height_curve
        if curve == "knee":
            # Smooth knee compression at ~knee_m; small heights ≈ linear, large heights compress
            # f(h) = h / (h + knee)
            mapped = Hm / (Hm + max(self.knee_m, 1e-6))
            # Optional gamma to lift shadows further (gamma<1 brightens small heights)
            if self.gamma is not None and self.gamma > 0:
                mapped = np.power(np.clip(mapped, 0.0, 1.0), self.gamma)
            norm = np.clip(mapped, 0.0, 1.0)

        elif curve == "gamma":
            # Robust max; then gamma curve (gamma < 1 brightens)
            vmax = float(np.percentile(Hm, self.vmax_percentile))
            vmax = max(vmax, 1e-6)
            base = np.clip(Hm / vmax, 0.0, 1.0)
            norm = np.power(base, max(self.gamma, 1e-6))

        elif curve == "log":
            # Log compression; log_gain controls steepness near zero
            vmax = float(np.percentile(Hm, self.vmax_percentile))
            vmax = max(vmax, 1e-6)
            k = max(self.log_gain, 1e-6)
            norm = np.log1p(k * Hm) / np.log1p(k * vmax)
            norm = np.clip(norm, 0.0, 1.0)
            if self.gamma is not None and self.gamma > 0:
                norm = np.power(norm, self.gamma)

        else:  # "linear"
            vmax = float(np.percentile(Hm, self.vmax_percentile))
            vmax = max(vmax, 1e-6)
            norm = np.clip(Hm / vmax, 0.0, 1.0)

        # Optional local contrast (CLAHE) to accentuate 1–7 m micro-relief
        if self.clahe:
            gray8 = (norm * 255.0).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile))
            norm = clahe.apply(gray8).astype(np.float32) / 255.0

        return norm.astype(np.float32)

    # ---------- snapshot & visualization ----------

    def _colorize(self, H: np.ndarray) -> np.ndarray:
        # """Return RGB uint8 colormap (FinderNet-like)."""
        # # Normalize robustly (ignore extreme zeros)
        # vmax = float(np.percentile(H, 99.0))
        # vmax = max(vmax, 1e-6)
        # norm = np.clip(H / vmax, 0.0, 1.0)
        # cmap = cm.get_cmap(self.colormap)
        # rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)  # drop alpha
        # return rgb
        """Return RGB uint8 colormap (FinderNet-like) with tone mapping."""
        norm = self._tone_map_heights(H)  # 0..1
        cmap = cm.get_cmap(self.colormap)
        rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)  # drop alpha
        return rgb
         

    def _edge_overlay(self, rgb: np.ndarray, H: np.ndarray) -> np.ndarray:
        # """Overlay edges extracted from DEM on top of color image."""
        # # work on 8-bit grayscale from H
        # vmax = float(np.percentile(H, 99.0)); vmax = max(vmax, 1e-6)
        # gray = np.clip(H / vmax, 0.0, 1.0)
        # gray8 = (gray * 255).astype(np.uint8)

        # edges = cv2.Canny(gray8, 50, 150)
        # # color edges (red)
        # overlay = rgb.copy()
        # r, g, b = overlay[..., 0], overlay[..., 1], overlay[..., 2]
        # r[edges > 0] = 255
        # g[edges > 0] = 0
        # b[edges > 0] = 0
        # return overlay
        """Overlay edges extracted from DEM on top of color image."""
        norm = self._tone_map_heights(H)
        gray8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        edges = cv2.Canny(gray8, 50, 150)
        overlay = rgb.copy()
        overlay[edges > 0] = (255, 0, 0)  # red
        return overlay

    def _shade_relief(self, rgb: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Simple hillshade-ish shading blended on top of colormap."""
        if H.max() < 1e-6:
            return rgb
        # Sobel gradients
        kx = cv2.Sobel(H, cv2.CV_32F, 1, 0, ksize=3)
        ky = cv2.Sobel(H, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(kx*kx + ky*ky)
        grad = grad / (grad.max() + 1e-6)
        shade = (0.7 + 0.3 * (1.0 - grad))  # brighter on flat, darker on steep
        out = (rgb.astype(np.float32) * shade[..., None]).clip(0, 255).astype(np.uint8)
        return out

    def _write_snapshot(self):
        """Write PNG (colorized, auto-cropped), raw .npy (full grid), and metadata JSON."""
        if self.grid is None:
            return

        # Filenames
        png = os.path.join(self.out_dir, f"dem_{self.snap_idx:04d}.png")
        npy = os.path.join(self.out_dir, f"dem_{self.snap_idx:04d}.npy")
        jso = os.path.join(self.out_dir, f"dem_{self.snap_idx:04d}_meta.json")

        # Keep full grid for downstream processing and reproducibility
        H_full = self.grid.copy()

        # ---- Auto-crop region for the PNG only ----
        mask = H_full > 0.0
        if mask.any():
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            margin = int(getattr(self, "crop_margin_px", 2))  # optional attr; default 24
            x0 = max(int(rows.min()) - margin, 0)
            x1 = min(int(rows.max()) + 1 + margin, H_full.shape[0])
            y0 = max(int(cols.min()) - margin, 0)
            y1 = min(int(cols.max()) + 1 + margin, H_full.shape[1])
        else:
            # Nothing rasterized yet → save full canvas
            x0, y0, x1, y1 = 0, 0, H_full.shape[0], H_full.shape[1]

        H = H_full[x0:x1, y0:y1]

        # ---- Colorize and overlays (on the cropped view) ----
        rgb = self._colorize(H)
        if self.shade:
            rgb = self._shade_relief(rgb, H)
        if self.overlay_edges:
            rgb = self._edge_overlay(rgb, H)

        # White background for empty pixels in the preview PNG
        if bool(getattr(self, "bg_white", True)):  # optional attr; default True
            rgb[H <= 0.0] = 255

        # ---- Write files ----
        imageio.imwrite(png, rgb)                    # cropped preview
        np.save(npy, H_full.astype(np.float32))      # full grid for accuracy

        # Provide mapping from cropped PNG back to world coords
        meta = dict(
            created_at=time.time(),
            resolution=self.resolution,
            xmin=self.xmin, ymin=self.ymin,
            grid_shape=[int(H_full.shape[0]), int(H_full.shape[1])],
            crop=dict(
                x0=int(x0), x1=int(x1), y0=int(y0), y1=int(y1),
                xmin_vis=float(self.xmin + x0 * self.resolution),
                ymin_vis=float(self.ymin + y0 * self.resolution),
            ),
            plane=self.plane.tolist(),
            R_align=self.R_align.tolist(),
            mode=self.mode,
            submaps_so_far=self.submap_count,
            colormap=self.colormap,
            overlay_edges=self.overlay_edges,
            shade=self.shade,
        )
        with open(jso, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[DEM] wrote {png}  (submaps={self.submap_count}, crop=({x0}:{x1},{y0}:{y1}))")
        self.snap_idx += 1
