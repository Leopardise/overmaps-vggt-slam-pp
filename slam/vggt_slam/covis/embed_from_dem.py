from __future__ import annotations
import numpy as np
import torch
import cv2

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

class DEMEmbedder:
    """
    Minimal DINOv2 wrapper that produces a single embedding vector from a
    uint8 grayscale patch (HxW). We replicate to 3 channels and normalize with
    ImageNet mean/std. Works with hub 'facebookresearch/dinov2' weights.
    """
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Map a friendly name to torch.hub identifiers
        # (dinov2-base → vitb14; large → vitl14; small → vits14)
        name_map = {
            "facebook/dinov2-small": "dinov2_vits14",
            "facebook/dinov2-base":  "dinov2_vitb14",
            "facebook/dinov2-large": "dinov2_vitl14",
            "facebook/dinov2-giant": "dinov2_vitg14",
        }
        hub_name = name_map.get(model_name, "dinov2_vitb14")

        # Load from torch.hub (will use your local cache if already present)
        self.model = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=True)
        self.model.eval().to(self.device)

        # expected embed dim per backbone
        self.embed_dim = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }[hub_name]

        # target spatial size (typical DINOv2 input)
        self.size = 224

    @torch.no_grad()
    def embed_uint8_patch(self, u8_gray: np.ndarray) -> np.ndarray:
        """
        u8_gray: HxW, uint8 (0..255). Returns (D,) float32 L2-normalized.
        """
        assert u8_gray.ndim == 2 and u8_gray.dtype == np.uint8, "expect HxW uint8"

        # Resize → replicate 3ch → float → normalize
        img = cv2.resize(u8_gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.stack([img, img, img], axis=-1)  # HxWx3
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)  # 1x3xHxW

        # Forward features
        out = self.model.forward_features(x)

        # Prefer class token if present, else mean of patch tokens
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                feats = out["x_norm_clstoken"]  # [1, D]
            elif "x_norm_cls" in out:
                feats = out["x_norm_cls"]      # [1, D]
            elif "x_norm_patchtokens" in out:
                feats = out["x_norm_patchtokens"].mean(dim=1)  # [1, D]
            elif "penultimate_layer" in out:
                feats = out["penultimate_layer"]
            else:
                # last resort: try to pool anything that looks like tokens
                toks = None
                for k in out:
                    v = out[k]
                    if torch.is_tensor(v) and v.ndimension() == 3 and v.shape[0] == 1:
                        toks = v; break
                feats = toks.mean(dim=1) if toks is not None else None
        else:
            feats = out

        if feats is None:
            raise RuntimeError("DINOv2 forward produced no usable features.")

        vec = feats.squeeze(0).detach().float().cpu().numpy()
        # L2 normalize for cosine/IP
        n = np.linalg.norm(vec) + 1e-12
        vec = (vec / n).astype(np.float32)
        return vec
