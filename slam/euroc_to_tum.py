#!/usr/bin/env python3

import argparse
import csv

def convert_euroc_to_tum(input_csv, output_txt):
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        lines_out = []
        for row in reader:
            if not row or row[0].startswith("#") or len(row) < 8:
                continue
            timestamp = int(row[0])
            px = float(row[1])
            py = float(row[2])
            pz = float(row[3])
            qw = float(row[4])
            qx = float(row[5])
            qy = float(row[6])
            qz = float(row[7])
            # Convert nanosecond timestamp to seconds
            ts_sec = timestamp * 1e-9
            lines_out.append(f"{ts_sec:.6f} {px:.6f} {py:.6f} {pz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    with open(output_txt, 'w') as fout:
        fout.write("\n".join(lines_out) + "\n")

    print(f"Converted {len(lines_out)} entries to TUM format → {output_txt}")

def main():
    parser = argparse.ArgumentParser(description="Convert EuRoC GT CSV to TUM format")
    parser.add_argument("--in", dest="input", required=True, help="Path to EuRoC GT CSV")
    parser.add_argument("--out", dest="output", required=True, help="Output TUM file")
    args = parser.parse_args()
    convert_euroc_to_tum(args.input, args.output)

if __name__ == "__main__":
    main()
