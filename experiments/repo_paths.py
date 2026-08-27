"""Repo-relative defaults. Override with env vars; do not hard-code host paths."""
from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[1]

SLAM_DIR = Path(os.environ.get("VGGT_SLAM_DIR", REPO / "slam"))
PYTHON = os.environ.get("PYTHON", sys.executable)
VGGT_CKPT = os.environ.get("VGGT_CKPT", "")

EVAL6 = REPO / "data" / "OverMaps-eval6"
INSTA = Path(os.environ.get("INSTA360_DIR", REPO / "data" / "insta360"))
POSES = REPO / "results" / "poses"
OUTPUTS = Path(os.environ.get("VGGT_OUT", REPO / "results" / "run"))


def overmaps_1k():
    p = os.environ.get("OVERMAPS_ROOT")
    if p:
        return Path(p)
    scratch = Path("/data1/avilasha2/overmaps/OverMaps-1K")
    if scratch.is_dir():
        return scratch
    return None


def eval6_dir(uuid, folder):
    return EVAL6 / f"{folder}__{uuid[:8]}"


def images_dir(uuid, folder):
    root = overmaps_1k()
    if root is not None:
        return root / "images" / uuid
    return eval6_dir(uuid, folder) / "images"


def sparse_dir(uuid, folder):
    root = overmaps_1k()
    if root is not None:
        return root / "sparse" / uuid / "0"
    return eval6_dir(uuid, folder) / "sparse" / "0"
