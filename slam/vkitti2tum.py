import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

def convert_vkitti_to_tum(input_path, output_path):
    # Try to read the file, skip header if present
    df = pd.read_csv(input_path, sep=r"\s+", comment="#", header=None)
    
    # Drop header row if first entry is not numeric
    if not str(df.iloc[0, 0]).replace('.', '', 1).isdigit():
        df = df.drop(0).reset_index(drop=True)

    if df.shape[1] < 17:
        raise ValueError("Unexpected file format — expected frame + 16 matrix entries.")

    with open(output_path, "w") as f_out:
        for _, row in df.iterrows():
            frame = int(float(row[0]))
            M = np.array([
                [row[1], row[2], row[3], row[4]],
                [row[5], row[6], row[7], row[8]],
                [row[9], row[10], row[11], row[12]],
                [0, 0, 0, 1]
            ], dtype=np.float64)

            # Convert world→camera to camera→world
            R_wc = M[:3, :3].T
            t_wc = -R_wc @ M[:3, 3]

            # VKITTI (x-right, y-down, z-forward) → TUM (x-right, y-up, z-backward)
            flip = np.diag([1, -1, -1])
            R_wc = flip @ R_wc
            t_wc = flip @ t_wc

            # Quaternion (x, y, z, w)
            q = R.from_matrix(R_wc).as_quat()
            qx, qy, qz, qw = q[0], q[1], q[2], q[3]

            # TUM format line
            timestamp = frame / 10.0  # adjust if frame rate differs
            f_out.write(f"{timestamp:.6f} {t_wc[0]:.6f} {t_wc[1]:.6f} {t_wc[2]:.6f} "
                        f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    print(f"✅ Converted {input_path} → {output_path} (TUM format)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vkitti2tum.py <input_txt> <output_tum>")
        sys.exit(1)
    convert_vkitti_to_tum(sys.argv[1], sys.argv[2])