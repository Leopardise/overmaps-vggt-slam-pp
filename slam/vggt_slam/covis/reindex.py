from __future__ import annotations
import os, shutil
from typing import Dict, Tuple, Optional
import numpy as np
from .covis_graph import CovisGraph, NodePayload
from vggt_slam.dem.patch_db import load_latest_dem_grid, crop_dem_patch_from_bbox, normalize_patch_for_matching
from .embed_from_dem import DEMEmbedder

def rebuild_covis_from_dem(old_covis: CovisGraph, new_run_root: str,
                           snapshot_every: int = 5,
                           model_name: str = "facebook/dinov2-base") -> CovisGraph:
    """
    Create a fresh CovisGraph (new_run_root) by embedding DEM patches of all nodes.
    Copies nothing automatically; you should have updated pcds/poses in new_run_root already.
    """
    os.makedirs(new_run_root, exist_ok=True)
    new_graph = CovisGraph(
        run_root=new_run_root,
        topk_edges=old_covis.topk_edges,
        snapshot_every=snapshot_every,
        hnsw_m=old_covis.hnsw_m, hnsw_efc=old_covis.hnsw_efc, hnsw_efs=old_covis.hnsw_efs
    )
    dem = load_latest_dem_grid(new_run_root)
    if dem is None:
        # fallback: try from old run root
        dem = load_latest_dem_grid(old_covis.run_root)

    embedder = DEMEmbedder(model_name=model_name)

    # keep ordering stable
    for sid in sorted(old_covis.nodes.keys()):
        p = old_covis.nodes[sid]
        # Use same pcd path basename but under new run_root/submaps
        submaps_dir = os.path.join(new_run_root, "submaps")
        os.makedirs(submaps_dir, exist_ok=True)
        pcd_path = os.path.join(submaps_dir, os.path.basename(p.pointcloud_path))

        # DEM bbox might shift slightly after pose updates, but we’ll reuse it; your DEM manager snapshot should be new.
        bbox = p.dem_patch_bbox
        if dem is not None and bbox is not None:
            patch = crop_dem_patch_from_bbox(dem, bbox)
            patch_u8 = normalize_patch_for_matching(patch)
            emb = embedder.embed_uint8_patch(patch_u8)
        else:
            # empty embedding (rare)
            emb = np.zeros((768,), np.float32)

        new_graph.add_submap(
            submap_id=sid,
            frames_uint8_chw=np.empty((0,)),  # not used by add_submap
            embedding=emb,
            pose_world=p.world_pose,
            pointcloud_path=pcd_path
        )

        if (sid % new_graph.snapshot_every) == 0:
            # you already call draw_graph_png from Solver; optional here
            pass

    # Save rebuilt index + metadata
    new_graph.save()
    return new_graph
