"""Render a trained gsplat splat along a Habitat agent camera path -> photoreal
first-person video. The decoupled "Habitat-GS" alternative: physics/nav from the
mesh (Habitat), pixels from the splat (gsplat). Maps each Habitat camera pose back
to the splat frame using the source->Habitat transform (scene.json M_s2h) and the
OpenGL->OpenCV camera convention flip.

Run in the `splat` env:
    python scripts/gsplat_render_path.py --ckpt ckpt.pt --scene scene.json \
        --path path.json --out explore_photoreal.mp4
"""

from __future__ import annotations

import argparse
import json
import numpy as np


def habitat_K(hfov_deg, width, height):
    hfov = np.deg2rad(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov / 2.0)
    return np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1.0]], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", required=True)        # scene.json with M_s2h
    ap.add_argument("--path", required=True)          # path.json from Habitat
    ap.add_argument("--out", required=True)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    import torch
    import imageio.v2 as imageio
    from gsplat import rasterization

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device)
    sp = ck["splats"] if "splats" in ck else ck
    means = sp["means"].to(device)
    scales = torch.exp(sp["scales"]).to(device)
    quats = sp["quats"].to(device)
    op = sp["opacities"].to(device)
    opacities = torch.sigmoid(op.squeeze(-1) if op.dim() > 1 else op)
    colors = torch.cat([sp["sh0"], sp["shN"]], dim=1).to(device)

    M = np.array(json.load(open(args.scene))["M_s2h"], dtype=np.float64)
    Minv = np.linalg.inv(M)
    GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])

    path = json.load(open(args.path))
    poses, hfov, W, H = path["poses"], path["hfov"], int(path["width"]), int(path["height"])
    K = torch.from_numpy(habitat_K(hfov, W, H)).to(device)[None]
    print(f"{len(poses)} poses, {W}x{H}, hfov {hfov}", flush=True)

    frames = []
    with torch.no_grad():
        for c2w_h in poses:
            c2w_s = Minv @ np.array(c2w_h, dtype=np.float64)   # Habitat world -> splat frame
            c2w_s_cv = c2w_s @ GL2CV                            # OpenGL cam -> OpenCV cam
            viewmat = torch.from_numpy(np.linalg.inv(c2w_s_cv)).float().to(device)[None]
            out, _, _ = rasterization(
                means, quats, scales, opacities, colors, viewmat, K, W, H,
                sh_degree=args.sh_degree, render_mode="RGB",
                near_plane=0.01, far_plane=1e10, packed=True, rasterize_mode="classic")
            arr = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            h, w = arr.shape[:2]
            frames.append(arr[: h - (h % 2), : w - (w % 2)])   # even dims for h264
    imageio.mimwrite(args.out, frames, fps=args.fps, quality=8, macro_block_size=1)
    print(f"wrote {args.out} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
