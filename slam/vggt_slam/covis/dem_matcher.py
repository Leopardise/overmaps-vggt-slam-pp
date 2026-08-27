from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .covis_graph import CovisGraph
from vggt_slam.dem.patch_db import load_latest_dem_meta, load_latest_dem_grid, crop_dem_patch_from_bbox, normalize_patch_for_matching, ncc_similarity

def rank_window_by_dem(covis: CovisGraph, current_id: int, window_ids: List[int], top_k: int) -> List[Tuple[int,float]]:
    """
    For the current node, extract its DEM patch; compare with patches from window_ids by NCC on normalized patches.
    Return top_k list of (id, score) in descending order (excluding current_id).
    """
    dem = load_latest_dem_grid(covis.run_root)
    if dem is None:
        return []
    cur_payload = covis.nodes.get(current_id, None)
    if cur_payload is None or cur_payload.dem_patch_bbox is None:
        return []
    cur_patch = crop_dem_patch_from_bbox(dem, cur_payload.dem_patch_bbox)
    cur_n = normalize_patch_for_matching(cur_patch)

    scored = []
    for sid in window_ids:
        if sid == current_id:
            continue
        p = covis.nodes.get(sid, None)
        if p is None or p.dem_patch_bbox is None:
            continue
        tgt_patch = crop_dem_patch_from_bbox(dem, p.dem_patch_bbox)
        tgt_n = normalize_patch_for_matching(tgt_patch)
        sc = ncc_similarity(cur_n, tgt_n)
        scored.append((sid, sc))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
