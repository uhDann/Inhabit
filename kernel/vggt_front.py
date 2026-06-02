"""VGGT multi-view feed-forward front end (pretrained facebook/VGGT-1B).

Predicts jointly-consistent depth + per-pixel confidence for a set of frames in one
pass. Scale-invariant, so a single global metric scale is applied downstream. Frozen
foundation model; the work is feeding its depth+confidence into the from-scratch kernel.
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image

_M = None


def _load():
    global _M
    if _M is None:
        from vggt.models.vggt import VGGT
        _M = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    return _M


@torch.no_grad()
def predict_depths(rgb_list, H, W):
    """rgb_list: list of HxWx3 uint8 (any size). Returns depth[N,H,W], conf[N,H,W]
    at the requested (H,W), scale-invariant."""
    m = _load(); dev = "cuda"
    ims = []
    for im in rgb_list:
        r = np.asarray(Image.fromarray(im).resize((W, H), Image.BILINEAR))[:, :, :3].copy()
        ims.append(torch.from_numpy(r).float() / 255.0)
    x = torch.stack(ims).permute(0, 3, 1, 2).to(dev)        # [N,3,H,W]
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        p = m(x)
    depth = p["depth"][0, ..., 0].float().cpu().numpy()     # [N,H,W]
    conf = p["depth_conf"][0].float().cpu().numpy()         # [N,H,W]
    del p, x
    torch.cuda.empty_cache()
    return depth, conf
