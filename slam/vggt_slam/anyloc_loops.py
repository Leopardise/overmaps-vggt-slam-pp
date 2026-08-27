#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, time
from typing import Dict, List, Tuple, Set

import numpy as np
from termcolor import colored

# --- helpers ---------------------------------------------------------------

def _norm_id(x) -> int:
    """
    Accepts: 3, "3", "sm_00003", "SM_12" -> 3 / 12
    """
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s.lower().startswith("sm_"):
        s = s[3:]
    # tolerate zero padding
    return int(s.lstrip("0") or "0")

def _read_votes_csv(path: str) -> List[Tuple[int, int, float, int]]:
    """
    Returns list of (query, match, score, rank) where rank∈{1,2}.
    Accepts headers: query_submap, match1_submap, match1_score, match2_submap, match2_score.
    Ignores rows missing either match.
    """
    out = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            q = row.get("query_submap")
            m1 = row.get("match1_submap")
            s1 = row.get("match1_score")
            m2 = row.get("match2_submap")
            s2 = row.get("match2_score")
            if q and m1:
                try:
                    out.append((_norm_id(q), _norm_id(m1), float(s1) if s1 else 0.0, 1))
                except Exception:
                    pass
            if q and m2:
                try:
                    out.append((_norm_id(q), _norm_id(m2), float(s2) if s2 else 0.0, 2))
                except Exception:
                    pass
    # Highest score first (tie-break: rank1 before rank2)
    out.sort(key=lambda t: (-t[2], t[3], t[0], t[1]))
    return out

# --- streamer --------------------------------------------------------------

class AnyLocStream:
    """
    Mid-run loop-closure injector driven by a CSV of submap matches.
    - Called after EACH submap commit.
    - If both submaps (q,m) exist, adds a between-factor immediately and optimizes.
    """

    def __init__(self, csv_path: str, use_sim3: bool = False, max_edges_per_tick: int = 4):
        self.csv_path = csv_path
        self.use_sim3 = bool(use_sim3)
        self.max_edges_per_tick = int(max_edges_per_tick)
        self._mtime = 0.0
        self._rows: List[Tuple[int,int,float,int]] = []
        self._added_edges: Set[Tuple[int,int]] = set()  # undirected key: (min,max)

    # public ----------------------------------------------------------------
    def on_submap_committed(self, solver, new_sm_id: int) -> int:
        """
        Returns how many edges were injected this tick.
        """
        if not self.csv_path or not os.path.isfile(self.csv_path):
            return 0

        # Reload CSV if changed
        mtime = os.path.getmtime(self.csv_path)
        if mtime > self._mtime or not self._rows:
            self._rows = _read_votes_csv(self.csv_path)
            self._mtime = mtime

        formed_ids = set(solver.map.get_all_submap_ids())  # e.g., [0..k]
        injected = 0

        for (q, m, score, rank) in self._rows:
            if injected >= self.max_edges_per_tick:
                break

            if q not in formed_ids or m not in formed_ids:
                continue

            key = (min(q, m), max(q, m))
            if key in self._added_edges:
                continue  # already added

            # build relative transform for factor
            if self.use_sim3:
                # Pose3 path: get poses of the "reference" frames and compute between
                pose_w_q = solver.map.get_submap(q).get_pose_reference_frame()  # 4x4
                pose_w_m = solver.map.get_submap(m).get_pose_reference_frame()  # 4x4
                # between: m -> q  (factor is Between(m,q))
                rel = np.linalg.inv(pose_w_m) @ pose_w_q
                solver.graph.add_between_factor(m, q, rel, solver.graph.relative_noise)
                print(colored(f"[anyloc] LC(Pose3): added {m} ↔ {q} (rank={rank}, score={score:.1f})", "yellow"))
            else:
                # SL(4) path: use current global homographies
                H_w_q = solver.map.get_submap(q).get_reference_homography()   # 4x4
                H_w_m = solver.map.get_submap(m).get_reference_homography()   # 4x4
                rel = np.linalg.inv(H_w_m) @ H_w_q
                solver.graph.add_between_factor(m, q, rel, solver.graph.relative_noise)
                print(colored(f"[anyloc] LC(SL4): added {m} ↔ {q} (rank={rank}, score={score:.1f})", "yellow"))

            self._added_edges.add(key)
            injected += 1

            # optimize immediately so you see the effect in logs/viser
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

        return injected
