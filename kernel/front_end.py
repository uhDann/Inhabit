"""Feed-forward metric-depth front end (pretrained, NOT trained from scratch).

Depth-Anything-V2 metric (indoor), used via the model API directly (the pipeline
helper is buggy on torch 2.2). Frozen foundation model; our work is the integration
into the from-scratch fusion kernel.
"""
from __future__ import annotations
import numpy as np
import torch

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
_M = None
_P = None


def _load():
    global _M, _P
    if _M is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _P = AutoImageProcessor.from_pretrained(MODEL)
        _M = AutoModelForDepthEstimation.from_pretrained(MODEL).to(dev).eval()
    return _M, _P


@torch.no_grad()
def predict_depth(rgb_uint8):
    """rgb_uint8: HxWx3 numpy. Returns metric depth (HxW float32, metres)."""
    from PIL import Image
    m, p = _load()
    dev = next(m.parameters()).device
    H, W = rgb_uint8.shape[:2]
    inp = p(images=Image.fromarray(rgb_uint8), return_tensors="pt").to(dev)
    out = m(**inp)
    d = out.predicted_depth                     # [1,h,w], metric metres
    d = torch.nn.functional.interpolate(d[:, None], size=(H, W), mode="bilinear",
                                        align_corners=False)[0, 0]
    return d.float().cpu().numpy()
