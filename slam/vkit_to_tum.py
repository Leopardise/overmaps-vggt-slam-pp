#!/usr/bin/env python3

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_vkit_line(line):
    parts = line.strip().split()
    if len(parts) < 13:
        return None
    frame = int(parts[0])
    x = float(parts[7])
    y = float(parts[8])
    z = float(parts[9])
    yaw = float(parts[10])  # rotation_world_space_y
    return frame, x, y, z, yaw

def yaw_to_quaternion(yaw):
    # Rotation only around Y-axis (up) in world coordinates
    r = R.from_euler('y', yaw)
    return r.as_quat()  # x, y, z, w

def convert_to_tum(input_path, output_path, time_scale=0.1):
    lines_out = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("frame"):
                continue

            parsed = parse_vkit_line(line)
            if parsed is None:
                continue
            frame, x, y, z, yaw = parsed
            timestamp = frame * time_scale
            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            lines_out.append(f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    with open(output_path, 'w') as f_out:
        f_out.write("\n".join(lines_out) + "\n")
    print(f"Converted {len(lines_out)} frames to TUM format → {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Virtual KITTI GT to TUM format")
    parser.add_argument("--in", dest="input", required=True, help="Input VKITTI ground truth file")
    parser.add_argument("--out", dest="output", required=True, help="Output TUM file")
    parser.add_argument("--dt", type=float, default=0.1, help="Time per frame (default: 0.1s)")
    args = parser.parse_args()
    convert_to_tum(args.input, args.output, args.dt)

if __name__ == "__main__":
    main()
