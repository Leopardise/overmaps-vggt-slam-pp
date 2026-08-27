# vggt_slam/loop_closure.py
"""RGB image-retrieval used only when --max_loops > 0.

VGGT-SLAM++ (paper Sec. 3) sets w_loops = 0 and performs loop detection
on DEM tiles/chips with DINOv2 + AnyLoc, not on RGB SALAD descriptors.
This module therefore defaults to a no-op retriever. Optional SALAD
support is kept for the original VGGT-SLAM front-end path.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional

import numpy as np
import torch


class LoopMatch(NamedTuple):
    similarity_score: float
    query_submap_id: int
    query_submap_frame: int
    detected_submap_id: int
    detected_submap_frame: int


class LoopMatchQueue:
    def __init__(self, max_size: int):
        self.max_size = max(1, int(max_size)) if max_size else 1
        self.items: List[LoopMatch] = []

    def add(self, match: LoopMatch):
        self.items.append(match)
        self.items.sort(key=lambda m: -float(m.similarity_score))
        self.items = self.items[: self.max_size]

    def get_matches(self) -> List[LoopMatch]:
        return list(self.items)


class ImageRetrieval:
    """Default no-op retriever. Pass enable_salad=True to load SALAD."""

    def __init__(self, input_size: int = 224, enable_salad: bool = False):
        self.input_size = input_size
        self.enable_salad = bool(enable_salad)
        self.model = None
        if self.enable_salad:
            self._load_salad()

    def _load_salad(self):
        import os
        import torchvision.transforms as T
        from salad.eval import load_model

        try:
            torch.hub.load("serizba/salad", "dinov2_salad")
        except Exception:
            pass
        ckpt = os.path.join(torch.hub.get_dir(), "checkpoints/dino_salad.ckpt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(ckpt).to(device).eval()
        self.device = device
        self.transform = T.Compose(
            [
                T.Resize((self.input_size, self.input_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.to_pil = T.ToPILImage()

    def get_all_submap_embeddings(self, submap):
        if not self.enable_salad or self.model is None:
            return []
        frames = submap.get_all_frames()
        return self.get_batch_descriptors(frames)

    @torch.no_grad()
    def get_batch_descriptors(self, imgs):
        batch = []
        if isinstance(imgs, (np.ndarray, torch.Tensor)) and getattr(imgs, "ndim", 0) == 4:
            frames_iter = (imgs[i] for i in range(imgs.shape[0]))
        else:
            frames_iter = imgs
        for img in frames_iter:
            if isinstance(img, np.ndarray):
                t = torch.from_numpy(img)
            else:
                t = img
            if t.ndim == 3 and t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
                t = t.permute(2, 0, 1)
            if t.dtype.is_floating_point:
                t = (t.clamp(0, 1) * 255).to(torch.uint8)
            batch.append(self.transform(self.to_pil(t)))
        x = torch.stack(batch, 0).to(self.device)
        return self.model(x)

    def find_loop_closures(self, map, submap, max_loop_closures: int = 0, max_similarity_thres: float = 0.80):
        if (not self.enable_salad) or self.model is None or max_loop_closures <= 0:
            return []
        matches = LoopMatchQueue(max_size=max_loop_closures)
        query_vecs = submap.get_all_retrieval_vectors()
        if query_vecs is None or (hasattr(query_vecs, "__len__") and len(query_vecs) == 0):
            return []
        query_id = submap.get_id()
        for qi, qv in enumerate(query_vecs):
            best_score, best_sid, best_fid = map.retrieve_best_score_frame(
                qv, query_id, ignore_last_submap=True
            )
            # SALAD uses L2 distance; convert to a similarity-like score
            sim = float(1.0 / (1.0 + best_score))
            if sim >= max_similarity_thres:
                matches.add(
                    LoopMatch(
                        similarity_score=sim,
                        query_submap_id=query_id,
                        query_submap_frame=qi,
                        detected_submap_id=int(best_sid),
                        detected_submap_frame=int(best_fid),
                    )
                )
        return matches.get_matches()
