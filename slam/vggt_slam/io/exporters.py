import os, csv, numpy as np
from pathlib import Path

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def save_submap_world_points(submap, out_dir):
    """
    Saves a single fused point set per submap, already in WORLD frame.
    Format: SUBMAP_####.npy with float32 (N,3).
    """
    ensure_dir(out_dir)
    # Use confidence-masked fusion already implemented by Submap.get_points_in_world_frame()
    pts_world = submap.get_points_in_world_frame().astype(np.float32)
    sid = submap.get_id()
    np.save(os.path.join(out_dir, f"SUBMAP_{sid:04d}.npy"), pts_world)
    return pts_world.shape[0]

def append_submap_pose_csv(submap, out_csv):
    """
    One line per submap: submap_id,tx,ty,tz,qx,qy,qz,qw
    We derive a representative cam2world from submap (use first frame).
    """
    ensure_dir(os.path.dirname(out_csv))
    # Representative global transform for the submap is its reference homography H_world_map
    H = submap.get_reference_homography()
    # Pose as translation + quaternion (from 3x3)
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    t = H[:3, 3]
    q = R.from_matrix(H[:3, :3]).as_quat()  # x,y,z,w
    header_needed = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["submap_id","tx","ty","tz","qx","qy","qz","qw"])
        w.writerow([submap.get_id(), float(t[0]), float(t[1]), float(t[2]),
                    float(q[0]), float(q[1]), float(q[2]), float(q[3])])
