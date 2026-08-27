from setuptools import setup, find_packages

setup(
    name="vggt_slam_pp",
    version="1.0.0",
    description="VGGT-SLAM++: transformer odometry with a DEM-based spatially corrective back-end",
    packages=find_packages(include=["evals", "evals.*", "vggt_slam", "vggt_slam.*"]),
)
