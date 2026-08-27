#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import csv
from typing import Dict, List, Tuple, Set
import numpy as np
from termcolor import colored
from vggt_slam.h_solve import ransac_projective

try:
    from vggt_slam.graph_se3 import estimate_sim3_umeyama as _estimate_sim3
except Exception:
    _estimate_sim3 = None  # guarded

class AnyLocStreamer:
    """
    On every committed submap K:
      - find CSV rows (query==K or detected==K) whose counterpart is already formed
      - add the edge immediately, optimize, and refresh viewer
    De-duped; sorted by score so best edges go first.
    """

    def __init__(self, csv_path: str, use_sim3: bool):
        self.csv_path = csv_path
        self.use_sim3 = bool(use_sim3)
        self._by_query: Dict[int, List[dict]] = {}
        self._by_detected: Dict[int, List[dict]] = {}
        self._inserted: Set[Tuple[int,int,int,int]] = set()
        self._loaded = False

    def _load_csv_once(self):
        if self._loaded:
            return
        self._loaded = True
        if not self.csv_path:
            return
        by_q: Dict[int, List[dict]] = {}
        by_d: Dict[int, List[dict]] = {}
        with open(self.csv_path, "r") as f:
            rd = csv.DictReader(f)
            for r in rd:
                try:
                    qsm = int(r["query_sm_id"])
                    dsm = int(r["detected_sm_id"])
                    qf  = int(r.get("query_frame", 0))
                    df  = int(r.get("detected_frame", 0))
                    score = float(r.get("score", 1e9))
                except Exception:
                    continue
                row = {"query_sm_id": qsm, "detected_sm_id": dsm,
                       "query_frame": qf, "detected_frame": df, "score": score}
                by_q.setdefault(qsm, []).append(row)
                by_d.setdefault(dsm, []).append(row)
        for d in (by_q, by_d):
            for k in d.keys():
                d[k].sort(key=lambda x: x["score"])
        self._by_query = by_q
        self._by_detected = by_d

    def _edge_key(self, row: dict) -> Tuple[int,int,int,int]:
        return (int(row["query_sm_id"]), int(row["query_frame"]),
                int(row["detected_sm_id"]), int(row["detected_frame"]))

    def _have_edge(self, row: dict) -> bool:
        return self._edge_key(row) in self._inserted

    def _mark_edge(self, row: dict):
        self._inserted.add(self._edge_key(row))

    def _sim3_between(self, solver, qsm: int, qf: int, dsm: int, df: int) -> np.ndarray:
        gmap = solver.get_graph_map()
        Pw_d = gmap.get_submap(dsm).get_frame_pointcloud(df).reshape(-1, 3)
        Pw_q = gmap.get_submap(qsm).get_frame_pointcloud(qf).reshape(-1, 3)
        if _estimate_sim3 is None:
            raise RuntimeError("estimate_sim3_umeyama not available from graph_se3.")
        T_rel, s = _estimate_sim3(Pw_d, Pw_q, use_robust=True)
        print(colored("scale factor", "green"), float(s))
        return np.asarray(T_rel, dtype=np.float64)

    def _sl4_between(self, solver, qsm: int, qf: int, dsm: int, df: int) -> np.ndarray:
        gmap = solver.get_graph_map()
        Pw_d = gmap.get_submap(dsm).get_frame_pointcloud(df).reshape(-1, 3)
        Pw_q = gmap.get_submap(qsm).get_frame_pointcloud(qf).reshape(-1, 3)
        H_rel = ransac_projective(Pw_q, Pw_d)  # detected→query
        return H_rel

    def _insert_edge_and_opt(self, solver, dsm: int, qsm: int, H_rel: np.ndarray):
        graph = solver.graph
        graph.add_between_factor(dsm, qsm, H_rel, graph.relative_noise)
        graph.increment_loop_closure()
        graph.optimize()
        solver.get_graph_map().update_submap_homographies(graph)
        try:
            solver.update_latest_submap_vis()
        except Exception:
            pass

    def _ready_rows_for_k(self, formed_ids: Set[int], k: int) -> List[dict]:
        rows = []
        rows.extend(self._by_query.get(k, []))
        rows.extend(self._by_detected.get(k, []))
        out = []
        for r in rows:
            qsm = int(r["query_sm_id"]); dsm = int(r["detected_sm_id"])
            if qsm in formed_ids and dsm in formed_ids and not self._have_edge(r):
                out.append(r)
        out.sort(key=lambda x: x["score"])
        return out

    def on_submap_committed(self, solver, new_sm_id: int, max_edges_now: int = 2):
        if not self.csv_path:
            return
        self._load_csv_once()
        gmap = solver.get_graph_map()
        formed_ids = set(sm.get_id() for sm in gmap.get_submaps())
        if new_sm_id not in formed_ids:
            return

        ready = self._ready_rows_for_k(formed_ids, new_sm_id)
        if not ready:
            return

        count = 0
        for row in ready:
            if count >= max_edges_now:
                break
            if self._have_edge(row):
                continue
            qsm = int(row["query_sm_id"]); dsm = int(row["detected_sm_id"])
            qf  = int(row["query_frame"]);  df  = int(row["detected_frame"])

            if self.use_sim3:
                H_rel = self._sim3_between(solver, qsm, qf, dsm, df)
            else:
                H_rel = self._sl4_between(solver, qsm, qf, dsm, df)

            self._insert_edge_and_opt(solver, dsm, qsm, H_rel)
            print(f"added loop closure factor {dsm} {qsm}")
            self._mark_edge(row)
            count += 1
