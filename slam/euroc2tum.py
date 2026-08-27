#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def load_euroc_csv(path: str) -> pd.DataFrame:
    """
    EuRoC ground-truth CSV header starts with '#timestamp,...'.
    DO NOT use comment='#' or you'll drop the header.
    """
    # Comma-separated, keep header as-is, allow leading spaces after commas.
    df = pd.read_csv(path, sep=",", engine="python", skip_blank_lines=True, skipinitialspace=True)
    # Drop fully-empty rows if any
    df = df.dropna(how="all")
    if df.empty:
        raise RuntimeError(f"No rows found in {path}")
    return df

def ensure_columns(df: pd.DataFrame, cols, what: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = (
            f"Missing {what} columns: {missing}\n"
            f"Available columns are:\n{list(df.columns)}\n\n"
            f"Tip: use --list-cols to inspect normalized names, or pass exact header names via --{what.replace(' ', '-')}"
        )
        raise ValueError(msg)

def compose_pose(order_tag: str, q0, q1, q2, q3, tx, ty, tz):
    """
    Build R (camera->world) and t (camera->world) from quaternion and translation
    given in CSV order specified by order_tag ('wxyz' or 'xyzw').
    """
    if order_tag == "wxyz":
        qw, qx, qy, qz = q0, q1, q2, q3
        q_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    elif order_tag == "xyzw":
        qx, qy, qz, qw = q0, q1, q2, q3
        q_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    else:
        raise ValueError("--quat-order must be 'wxyz' or 'xyzw'")
    R_cw = R.from_quat(q_xyzw).as_matrix()
    t_cw = np.array([tx, ty, tz], dtype=np.float64)
    return R_cw, t_cw

def maybe_invert(R_cw, t_cw, invert: bool):
    """
    If CSV pose is world->camera/body and you want camera->world (TUM), invert it.
    """
    if not invert:
        return R_cw, t_cw
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    return R_wc, t_wc

def normalize_col(name: str) -> str:
    """
    Helper only for --list-cols: present a readable, unit-stripped, lowercase view.
    """
    import re
    s = re.sub(r"\s*\[.*?\]\s*$", "", str(name))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def main():
    ap = argparse.ArgumentParser(description="Convert EuRoC-style CSV to TUM trajectory TXT.")
    ap.add_argument("input_csv", help="EuRoC CSV (e.g., .../state_groundtruth_estimate0/data.csv)")
    ap.add_argument("output_tum", help="Output TUM txt path")
    ap.add_argument("--invert", action="store_true",
                    help="Use if CSV stores world→body/camera and you want camera→world (TUM).")
    ap.add_argument("--quat-order", default="wxyz", choices=["wxyz", "xyzw"],
                    help="Order of quaternion columns in CSV. EuRoC GT is typically wxyz.")
    ap.add_argument("--time-scale", type=float, default=1e-9,
                    help="Multiply timestamp by this (default ns→s = 1e-9).")
    ap.add_argument("--time-offset", type=float, default=0.0,
                    help="Add offset (seconds) to timestamps.")
    ap.add_argument("--pos-cols", default="",
                    help="Override position columns as 'px,py,pz' (use EXACT header names, incl. units).")
    ap.add_argument("--quat-cols", default="",
                    help="Override quaternion columns as 'a,b,c,d' (EXACT header names, incl. units).")
    ap.add_argument("--list-cols", action="store_true",
                    help="Print columns and normalized preview, then exit.")
    args = ap.parse_args()

    df = load_euroc_csv(args.input_csv)

    if args.list_cols:
        print("Columns present in CSV (exact):")
        for c in df.columns:
            print(f"  {c}    (norm: {normalize_col(c)})")
        sys.exit(0)

    # Time column: EuRoC header is '#timestamp' in ns
    # Keep it literal; do not strip '#'
    time_col_candidates = ["#timestamp", "timestamp", "time", "t"]
    time_col = None
    for c in time_col_candidates:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # Fallback to the very first column
        time_col = df.columns[0]

    # Position columns (must be provided or detected)
    if args.pos_cols:
        px, py, pz = [s.strip() for s in args.pos_cols.split(",")]
        ensure_columns(df, [px, py, pz], "pos-cols")
    else:
        # Auto-detection for common EuRoC GT names with units
        candidates = [
            ("p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"),
            ("p_RS_R_x[m]",  "p_RS_R_y[m]",  "p_RS_R_z[m]"),
            ("p_x", "p_y", "p_z"),  # rare fallback
            ("x", "y", "z"),
        ]
        found = None
        for triple in candidates:
            if all(c in df.columns for c in triple):
                found = triple
                break
        if not found:
            raise ValueError(
                "Position columns not found. Pass them explicitly with --pos-cols "
                "using the EXACT header names (including units)."
            )
        px, py, pz = found

    # Quaternion columns (must be provided or detected)
    if args.quat_cols:
        qa, qb, qc, qd = [s.strip() for s in args.quat_cols.split(",")]
        ensure_columns(df, [qa, qb, qc, qd], "quat-cols")
        order_tag = args.quat_order
        c0, c1, c2, c3 = qa, qb, qc, qd
    else:
        # Auto-detection for EuRoC GT names (with units)
        # Typical: q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []
        q_wxyz = ("q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []")
        q_xyzw = ("q_RS_x []", "q_RS_y []", "q_RS_z []", "q_RS_w []")
        found = None
        order_tag = "wxyz"
        if all(c in df.columns for c in q_wxyz):
            found = q_wxyz; order_tag = "wxyz"
        elif all(c in df.columns for c in q_xyzw):
            found = q_xyzw; order_tag = "xyzw"
        else:
            # Try unitless variants as fallback
            q_wxyz2 = ("q_RS_w", "q_RS_x", "q_RS_y", "q_RS_z")
            q_xyzw2 = ("q_RS_x", "q_RS_y", "q_RS_z", "q_RS_w")
            if all(c in df.columns for c in q_wxyz2):
                found = q_wxyz2; order_tag = "wxyz"
            elif all(c in df.columns for c in q_xyzw2):
                found = q_xyzw2; order_tag = "xyzw"
        if not found:
            raise ValueError(
                "Quaternion columns not found. Pass them explicitly with --quat-cols "
                "using the EXACT header names (including units)."
            )
        c0, c1, c2, c3 = found
        # If user forced a different order, honor it
        if args.quat_order and args.quat_order != order_tag:
            order_tag = args.quat_order

    # Write TUM file
    with open(args.output_tum, "w") as f_out:
        for _, row in df.iterrows():
            # timestamp → seconds
            try:
                ts = float(row[time_col]) * args.time_scale + args.time_offset
            except Exception:
                # skip malformed rows
                continue

            # position
            tx = float(row[px]); ty = float(row[py]); tz = float(row[pz])
            # quaternion in CSV order
            q0 = float(row[c0]); q1 = float(row[c1]); q2 = float(row[c2]); q3 = float(row[c3])

            R_cw, t_cw = compose_pose(order_tag, q0, q1, q2, q3, tx, ty, tz)
            R_out, t_out = maybe_invert(R_cw, t_cw, args.invert)

            q_xyzw = R.from_matrix(R_out).as_quat()  # (x,y,z,w)
            qx, qy, qz, qw = q_xyzw

            f_out.write(
                f"{ts:.6f} {t_out[0]:.6f} {t_out[1]:.6f} {t_out[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )

    print(f"Converted {args.input_csv} → {args.output_tum} (TUM format)")

if __name__ == "__main__":
    main()
