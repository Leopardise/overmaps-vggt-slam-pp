from __future__ import annotations
import torch, numpy as np
from typing import List, Optional
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

_OPEN_DEFAULT = "facebook/dinov2-base"                         # public & ungated
_D3_EXAMPLE   = "facebook/dinov3-vitb16-pretrain-lvd1689m"     # gated; use if you get access

class DINOv3Embedder:
    """
    Produces L2-normalized global embeddings using a HuggingFace vision backbone.
    Defaults to DINOv2 (open). If a requested model is gated/unavailable, falls back to DINOv2.
    """
    def __init__(self,
                 model_name: str = _OPEN_DEFAULT,
                 dtype: str = "auto",
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            try:
                major_cc = torch.cuda.get_device_capability()[0]
                self.torch_dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
            except Exception:
                self.torch_dtype = torch.float16
        else:
            self.torch_dtype = getattr(torch, dtype)

        self.model_name = model_name
        self.processor = None
        self.model = None
        self._load_or_fallback()

    def _try_load(self, name: str):
        try:
            proc = AutoImageProcessor.from_pretrained(name)
        except Exception:
            proc = AutoProcessor.from_pretrained(name)
        mdl  = AutoModel.from_pretrained(name, torch_dtype=self.torch_dtype)
        mdl.eval().to(self.device)
        return proc, mdl

    def _load_or_fallback(self):
        try:
            self.processor, self.model = self._try_load(self.model_name)
        except Exception as e:
            print(f"[COVIS] '{self.model_name}' unavailable ({e}). Falling back to '{_OPEN_DEFAULT}'.")
            self.processor, self.model = self._try_load(_OPEN_DEFAULT)

    @torch.no_grad()
    def frames_to_submap_embedding(self, frames_uint8_chw: np.ndarray) -> np.ndarray:
        """
        frames_uint8_chw: (S, 3, H, W) uint8 [0..255]
        Returns (D,) float32 L2-normalized embedding (mean over frames).
        """
        feats: List[torch.Tensor] = []
        for f in frames_uint8_chw:
            img = Image.fromarray(np.transpose(f, (1, 2, 0)))
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            out = self.model(**inputs)

            if hasattr(out, "pooler_output") and (out.pooler_output is not None):
                emb = out.pooler_output  # [1, D]
            else:
                last = getattr(out, "last_hidden_state", None)
                if last is None:
                    raise RuntimeError("Model output lacks last_hidden_state/pooler_output.")
                emb = last[:, 0, :]  # CLS

            feats.append(emb.float().cpu())

        E = torch.stack(feats, dim=0).mean(dim=0).squeeze(0)  # [D]
        E = E / (E.norm(p=2) + 1e-12)
        return E.numpy().astype(np.float32)
