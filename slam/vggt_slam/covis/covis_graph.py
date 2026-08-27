from __future__ import annotations
import os, json, glob
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import faiss
import numpy as np
import networkx as nx

@dataclass
class NodePayload:
    submap_id: int
    pointcloud_path: str                                   # outputs/run_xx/submaps/SUBMAP_####.npy
    dem_patch_bbox: Optional[Tuple[int,int,int,int]]       # (x0,x1,y0,y1) in DEM grid indices
    world_pose: Tuple[float,float,float,float,float,float,float]  # (tx,ty,tz,qx,qy,qz,qw)
    embedding: np.ndarray = field(repr=False)              # (D,) L2-normalized
    embedding_path: Optional[str] = None                   # persisted vector for reuse
    updated_at: int = 0                                    # last step id when heavy repr was refreshed
    repr_version: int = 1                                  # bump if feature pipeline changes

class CovisGraph:
    """
    Covisibility graph with:
      - Node payloads: submap_id, world_pose, pointcloud path, DEM bbox, embedding (+ persisted)
      - Parent edge := nearest neighbor (cosine on L2-normed embeddings; FAISS HNSW)
      - Similarity edges := top-K neighbors
      - PNG snapshots via visualize.py (unchanged)
      - Persistence: FAISS index + nodes.json (+ embedding .npy files)
    """
    def __init__(self,
                 run_root: str = "outputs/run_01",
                 topk_edges: int = 5,
                 snapshot_every: int = 5,
                 hnsw_m: int = 32,
                 hnsw_efc: int = 100,
                 hnsw_efs: int = 64):
        self.run_root = run_root
        self.nodes: Dict[int, NodePayload] = {}
        self.G = nx.DiGraph()
        self.topk_edges = int(topk_edges)
        self.snapshot_every = int(snapshot_every)

        self.emb_dim: Optional[int] = None
        self.index: Optional[faiss.Index] = None

        # HNSW knobs
        self.hnsw_m = int(hnsw_m)
        self.hnsw_efc = int(hnsw_efc)
        self.hnsw_efs = int(hnsw_efs)

        self.covis_dir = os.path.join(self.run_root, "covis")
        os.makedirs(self.covis_dir, exist_ok=True)
        self.emb_dir = os.path.join(self.covis_dir, "embeddings")
        os.makedirs(self.emb_dir, exist_ok=True)
        self.snap_idx = 1

    # ---------- FAISS helpers ----------
    def _ensure_index(self, dim: int):
        if self.index is None:
            self.emb_dim = dim
            base = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efConstruction = self.hnsw_efc
            base.hnsw.efSearch = self.hnsw_efs
            self.index = faiss.IndexIDMap(base)

    def _add_to_index(self, submap_id: int, emb: np.ndarray):
        self._ensure_index(emb.shape[0])
        ids = np.array([submap_id], dtype=np.int64)
        self.index.add_with_ids(emb[None, :], ids)

    def _search_topk(self, emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores [k], ids [k]) (ids are submap_ids)."""
        if self.index is None or self.index.ntotal == 0:
            return np.array([]), np.array([])
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.hnsw_efs
        scores, ids = self.index.search(emb[None, :], min(k, self.index.ntotal))
        return scores[0], ids[0]

    # ---------- DEM bbox utility ----------
    def _load_latest_dem_meta(self) -> Optional[dict]:
        metas = sorted(glob.glob(os.path.join(self.run_root, "DEMs", "dem_*_meta.json")))
        if not metas:
            return None
        with open(metas[-1], "r") as f:
            return json.load(f)

    def _compute_dem_bbox_for_submap(self, pts_world: np.ndarray, dem_meta: dict) -> Optional[Tuple[int,int,int,int]]:
        """Project WORLD pts to DEM grid using dem_meta (R_align, xmin/ymin, resolution)."""
        try:
            R = np.array(dem_meta["R_align"], dtype=np.float64)
            res = float(dem_meta["resolution"])
            xmin = float(dem_meta["xmin"]); ymin = float(dem_meta["ymin"])
            Hshape = dem_meta["grid_shape"]  # [nx, ny]
            pts = (R @ pts_world.T).T
            x, y = pts[:,0], pts[:,1]
            gx = np.floor((x - xmin) / res).astype(np.int64)
            gy = np.floor((y - ymin) / res).astype(np.int64)
            valid = (gx >= 0) & (gy >= 0) & (gx < Hshape[0]) & (gy < Hshape[1])
            if not np.any(valid):
                return None
            gxv, gyv = gx[valid], gy[valid]
            return (int(gxv.min()), int(gxv.max()), int(gyv.min()), int(gyv.max()))
        except Exception:
            return None

    # ---------- Persistence ----------
    def save(self):
        """Persist FAISS index + node metadata."""
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.covis_dir, "faiss_index.bin"))
        meta = {
            str(sid): {
                "pointcloud_path": p.pointcloud_path,
                "dem_patch_bbox": p.dem_patch_bbox,
                "world_pose": p.world_pose,
                "emb_dim": int(p.embedding.shape[0]),
                "embedding_path": p.embedding_path,
                "updated_at": p.updated_at,
                "repr_version": p.repr_version,
            } for sid, p in self.nodes.items()
        }
        with open(os.path.join(self.covis_dir, "nodes.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load(self):
        """Restore FAISS index and node metadata (embeddings are not reloaded into RAM)."""
        idx_path = os.path.join(self.covis_dir, "faiss_index.bin")
        meta_path = os.path.join(self.covis_dir, "nodes.json")
        if os.path.exists(idx_path):
            self.index = faiss.read_index(idx_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for sid, d in meta.items():
                sid_i = int(sid)
                self.nodes[sid_i] = NodePayload(
                    submap_id=sid_i,
                    pointcloud_path=d["pointcloud_path"],
                    dem_patch_bbox=tuple(d["dem_patch_bbox"]) if d["dem_patch_bbox"] else None,
                    world_pose=tuple(d["world_pose"]),
                    embedding=np.zeros((d.get("emb_dim", 0),), dtype=np.float32),
                    embedding_path=d.get("embedding_path"),
                    updated_at=int(d.get("updated_at", 0)),
                    repr_version=int(d.get("repr_version", 1)),
                )
                self.G.add_node(
                    sid_i,
                    pcd=os.path.basename(d["pointcloud_path"]),
                    pose=self.nodes[sid_i].world_pose,
                    dem_bbox=self.nodes[sid_i].dem_patch_bbox
                )

    # ---------- Public API ----------
    def add_submap(self,
                   submap_id: int,
                   frames_uint8_chw: np.ndarray,
                   embedding: np.ndarray,
                   pose_world: Tuple[float,float,float,float,float,float,float],
                   pointcloud_path: str) -> Dict:
        """
        Insert a new node; assign parent via FAISS NN; create top-K similarity edges.
        Returns dict with parent, topk list and DEM bbox.
        """
        # 1) DEM bbox
        dem_bbox = None
        try:
            dem_meta = self._load_latest_dem_meta()
            if dem_meta and os.path.exists(pointcloud_path):
                pts = np.load(pointcloud_path)  # (N,3) WORLD
                dem_bbox = self._compute_dem_bbox_for_submap(pts, dem_meta)
        except Exception:
            dem_bbox = None

        # 2) Payload (+ persist embedding once)
        emb = embedding.astype(np.float32)
        emb_path = os.path.join(self.emb_dir, f"SUBMAP_{submap_id:04d}.npy")
        try:
            np.save(emb_path, emb)
        except Exception:
            emb_path = None  # non-fatal

        payload = NodePayload(
            submap_id=submap_id,
            pointcloud_path=pointcloud_path,
            dem_patch_bbox=dem_bbox,
            world_pose=pose_world,
            embedding=emb,
            embedding_path=emb_path,
        )

        # 3) Parent NN
        parent_id = None
        parent_score = None
        if len(self.nodes) > 0:
            scores, ids = self._search_topk(payload.embedding, k=self.topk_edges)
            valid = ids[ids >= 0]
            if valid.size > 0:
                parent_id = int(valid[0])
                parent_score = float(scores[0])

        # 4) Register node + FAISS
        self.nodes[submap_id] = payload
        self.G.add_node(submap_id, **{
            "pcd": os.path.basename(pointcloud_path),
            "pose": payload.world_pose,
            "dem_bbox": payload.dem_patch_bbox,
            "dim": payload.embedding.shape[0],
        })
        self._add_to_index(submap_id, payload.embedding)

        # 5) Similarity edges (top-K) + parent tag
        topk_edges: List[Tuple[int,int,float]] = []
        if len(self.nodes) > 1:
            scores, ids = self._search_topk(payload.embedding, k=self.topk_edges)
            for s_id, s in zip(ids, scores):
                s_id = int(s_id)
                if s_id < 0 or s_id == submap_id:
                    continue
                self.G.add_edge(s_id, submap_id, sim=float(s), kind="sim")
                topk_edges.append((s_id, submap_id, float(s)))
            if parent_id is not None:
                self.G.add_edge(parent_id, submap_id, sim=float(parent_score), kind="parent")

        return {
            "parent": parent_id,
            "parent_score": parent_score,
            "topk": topk_edges,
            "dem_bbox": dem_bbox,
        }

    def to_networkx(self) -> nx.DiGraph:
        return self.G

    # --------- Local window traversal (for loop detection later) ---------
    def collect_local_window(self, start_id: int,
                             max_nodes: int = 30,
                             max_hops: int = 2) -> List[int]:
        """
        BFS over parent + sim edges out to 'max_hops', capped at 'max_nodes'.
        """
        if start_id not in self.G:
            return []
        visited = {start_id}
        frontier = [start_id]
        hops = 0
        while frontier and len(visited) < max_nodes and hops < max_hops:
            nxt = []
            for u in frontier:
                preds = list(self.G.predecessors(u))
                succs = list(self.G.successors(u))
                nbrs = set(preds + succs)
                for v in nbrs:
                    if v not in visited:
                        visited.add(v)
                        if len(visited) >= max_nodes:
                            break
                        nxt.append(v)
                if len(visited) >= max_nodes:
                    break
            frontier = nxt
            hops += 1
        return list(visited)

    # --------- Selection for heavy updates inside a window ---------
    def top_similar_within(self, query_emb: np.ndarray, candidates: List[int], k: int) -> List[Tuple[int, float]]:
        """
        Return top-k (id, score) from `candidates` only, ranked by cosine similarity to `query_emb`.
        """
        if self.index is None or self.index.ntotal == 0 or not candidates:
            return []
        # get more than k then filter down
        scores, ids = self._search_topk(query_emb, k=max(k * 4, k + 8))
        cand_set = set(candidates)
        ranked: List[Tuple[int, float]] = []
        for sid, sc in zip(ids.tolist(), scores.tolist()):
            if sid in cand_set and sid not in [r[0] for r in ranked]:
                ranked.append((sid, sc))
                if len(ranked) >= k:
                    break
        return ranked

    def mark_updated(self, node_ids: List[int], step: int):
        """
        Bookkeeping: stamp nodes that had their heavy representation (DEM/pcd cache) refreshed.
        """
        for nid in node_ids:
            p = self.nodes.get(nid, None)
            if p is not None:
                p.updated_at = int(step)
