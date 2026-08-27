# vggt_slam/anyloc_csv.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, os
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from termcolor import colored

# CSV headers we expect (case-insensitive)
_HDR_Q  = "query_submap"
_HDR_M1 = "match1_submap"
_HDR_S1 = "match1_score"
_HDR_M2 = "match2_submap"
_HDR_S2 = "match2_score"

def _norm_id(x) -> Optional[int]:
    """
    Normalize submap ids:
      3 -> 3
      "3" -> 3
      "sm_00003" / "SM_12" -> 3 / 12
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower().startswith("sm_"):
        s = s[3:]
    s = s.lstrip("0") or "0"
    try:
        return int(s)
    except Exception:
        return None

def _float_or_zero(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def read_loop_votes(csv_path: str, topk_per_query: int = 1) -> List[Tuple[int,int,float,int]]:
    """
    Returns a ranked list of (q, m, score, rank) with highest score first.
    Keeps up to `topk_per_query` matches per query.
    Accepts headers: query_submap, match1_submap, match1_score, match2_submap, match2_score.
    """
    rows: List[Tuple[int,int,float,int]] = []
    if not (csv_path and os.path.isfile(csv_path)):
        return rows

    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            q = _norm_id(row.get(_HDR_Q))
            m1 = _norm_id(row.get(_HDR_M1))
            m2 = _norm_id(row.get(_HDR_M2))
            s1 = _float_or_zero(row.get(_HDR_S1))
            s2 = _float_or_zero(row.get(_HDR_S2))
            if q is None:
                continue
            if m1 is not None:
                rows.append((q, m1, s1, 1))
            if m2 is not None:
                rows.append((q, m2, s2, 2))

    # Sort by score desc; tiebreak by rank (1 before 2), then by ids for determinism
    rows.sort(key=lambda t: (-t[2], t[3], t[0], t[1]))

    # Keep only top-k per query
    topk: Dict[int, List[Tuple[int,int,float,int]]] = {}
    for q, m, s, r in rows:
        lst = topk.setdefault(q, [])
        if len(lst) < topk_per_query:
            lst.append((q, m, s, r))
    flat: List[Tuple[int,int,float,int]] = []
    for q in sorted(topk):
        flat.extend(topk[q])
    return flat

def add_loop_edges_from_csv(solver, csv_path: str,
                            max_edges: Optional[int] = None,
                            optimize_each: bool = True) -> int:
    """
    Adds loop-closure edges for any (q,m) present in CSV **and** already present in the map.
    Avoids duplicates. Uses SE3/Pose3 if solver.use_sim3 is True, else SL(4) homographies.
    Returns number of edges added.
    """
    if not csv_path or not os.path.isfile(csv_path):
        return 0

    pairs = read_loop_votes(csv_path, topk_per_query=1)
    if not pairs:
        return 0

    added = 0
    formed_ids: Set[int] = set(solver.map.submaps.keys())

    for q, m, score, rank in pairs:
        if max_edges is not None and added >= max_edges:
            break
        if q not in formed_ids or m not in formed_ids:
            continue

        key = (min(q, m), max(q, m))
        if key in solver._lc_already_added:
            continue

        if solver.use_sim3:
            # Use representative submap poses (world frame) to build SE3 between
            sm_q = solver.map.get_submap(q)
            sm_m = solver.map.get_submap(m)
            idx_q = sm_q.get_last_non_loop_frame_index()
            idx_m = sm_m.get_last_non_loop_frame_index()
            Pw_q = sm_q.get_pose_subframe(idx_q)   # 4x4 cam->world
            Pw_m = sm_m.get_pose_subframe(idx_m)   # 4x4 cam->world

            # Between: m -> q (factor Between(m,q))
            rel = np.linalg.inv(Pw_m) @ Pw_q
            solver.graph.add_between_factor(m, q, rel, solver.graph.relative_noise)
            print(colored(f"[loops] Pose3 LC: {m} ↔ {q} (rank={rank}, score={score:.2f})", "yellow"))
        else:
            # SL(4): use current world homographies
            H_w_q = solver.map.get_submap(q).get_reference_homography()  # 4x4 in SL(4)
            H_w_m = solver.map.get_submap(m).get_reference_homography()
            rel = np.linalg.inv(H_w_m) @ H_w_q
            solver.graph.add_between_factor(m, q, rel, solver.graph.relative_noise)
            print(colored(f"[loops] SL4  LC: {m} ↔ {q} (rank={rank}, score={score:.2f})", "yellow"))

        solver._lc_already_added.add(key)
        added += 1

        if optimize_each:
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

    return added
