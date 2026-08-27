from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def _nice_ticks(vmin: float, vmax: float, n: int = 6):
    # n ticks inclusive, rounded nicely
    if vmax <= vmin:
        return [vmin]
    vals = np.linspace(vmin, vmax, n)
    # round to 2 decimals (matches your examples), but keep -0.00 away
    out = []
    for x in vals:
        xr = float(np.round(x, 2))
        if abs(xr) < 0.005:
            xr = 0.0
        out.append(xr)
    return out


def render_height_wheel(index_json: str, out_path: str,
                        size_px: int = 900,
                        ring_inner: float = 0.55,
                        ring_outer: float = 0.98,
                        n_ticks: int = 6,
                        transparent: bool = False):
    index_json = Path(index_json)
    if not index_json.exists():
        raise FileNotFoundError(f"index.json not found: {index_json.resolve()}")

    with open(index_json, "r") as f:
        idx = json.load(f)

    viz_lo = float(idx.get("viz_lo", 0.0))
    viz_hi = float(idx.get("viz_hi", 1.0))

    # colormap name (prefer your explicit contract if present)
    cmap_name = "viridis"
    cm = idx.get("color_mapping", {})
    if isinstance(cm, dict):
        cmap_name = cm.get("colormap", cmap_name)

    cmap = mpl.cm.get_cmap(cmap_name)

    # Build a ring image: angle encodes height (viz_lo->viz_hi), radius just ring thickness
    N = int(size_px)
    yy, xx = np.mgrid[-1:1:complex(0, N), -1:1:complex(0, N)]
    rr = np.sqrt(xx**2 + yy**2)
    ang = (np.arctan2(yy, xx) + np.pi) / (2 * np.pi)  # [0,1)

    ring_mask = (rr >= ring_inner) & (rr <= ring_outer)

    # Normalize height by angle
    h_norm = ang  # 0..1
    rgba = cmap(h_norm)  # NxNx4
    # Outside ring -> transparent or white
    if transparent:
        rgba[..., 3] = np.where(ring_mask, rgba[..., 3], 0.0)
    else:
        # white background
        rgba[..., :3] = np.where(ring_mask[..., None], rgba[..., :3], 1.0)
        rgba[..., 3] = 1.0

    # Figure
    dpi = 200
    fig_size_in = (N / dpi, N / dpi)
    fig = plt.figure(figsize=fig_size_in, dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(rgba, extent=[-1, 1, -1, 1], interpolation="bilinear")
    ax.set_axis_off()
    ax.set_aspect("equal")

    # Title in center (optional but helps interpretation)
    ax.text(0, 0, "Height\n(m)", ha="center", va="center",
            fontsize=18, fontweight="semibold",
            color="black" if not transparent else "white",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=(1, 1, 1, 0.85) if not transparent else (0, 0, 0, 0.35),
                      edgecolor="none"))

    # Tick labels around circle
    ticks = _nice_ticks(viz_lo, viz_hi, n=n_ticks)
    # Place labels at fixed angles (top = max, clockwise decreases)
    # We want max at angle=90° (top). Our ang=0 is left? Actually atan2 with yy,xx:
    # ang=0 at (-1,0) (left), ang=0.25 at (0,1) (top). So top is 0.25.
    for i, tval in enumerate(ticks):
        frac = (tval - viz_lo) / (viz_hi - viz_lo + 1e-12)  # 0..1
        # map frac to angle so that frac=1 at top (0.25), frac=0 at just below top going clockwise
        # We’ll set theta = 0.25 - frac (wrap).
        theta = (0.25 - frac) * 2 * np.pi
        r_lab = 1.06
        x = r_lab * np.cos(theta)
        y = r_lab * np.sin(theta)

        # small tick mark on ring outer edge
        r0, r1 = ring_outer, ring_outer + 0.03
        x0, y0 = r0 * np.cos(theta), r0 * np.sin(theta)
        x1, y1 = r1 * np.cos(theta), r1 * np.sin(theta)
        ax.plot([x0, x1], [y0, y1], linewidth=2, color="black")

        # label
        ax.text(x, y, f"{tval:.2f} m", ha="center", va="center",
                fontsize=14, color="black",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=(1, 1, 1, 0.9),
                          edgecolor="none"))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: avoid cropping by giving pad + not relying on tight_layout
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15, transparent=transparent)
    plt.close(fig)

    print(f"[render_height_wheel] wrote: {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="path to index.json")
    ap.add_argument("--out", required=True, help="output PNG path (e.g., outputs/.../height_wheel.png)")
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--transparent", action="store_true")
    args = ap.parse_args()

    render_height_wheel(args.index, args.out, size_px=args.size, n_ticks=args.ticks, transparent=args.transparent)
