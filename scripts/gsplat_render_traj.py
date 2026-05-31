"""Render the splat along the actual capture trajectory (COLMAP training poses,
in order). These are all in-manifold, so every frame is sharp (unlike gsplat's
ellipse/interp paths that wander into unobserved regions). The reliable
photorealistic "fly through the reconstructed room" video.

Run in the `splat` env:
    python scripts/gsplat_render_traj.py --room-dir <room> --ckpt ckpt.pt \
        --gsplat-examples ~/gsplat-src/examples --out tour.mp4 --stride 2
"""

from __future__ import annotations

import argparse
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gsplat-examples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--compare", action="store_true", help="side-by-side: original photo | reconstruction")
    args = ap.parse_args()

    sys.path.insert(0, args.gsplat_examples)
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    from gsplat import rasterization
    from datasets.colmap import Parser

    parser = Parser(data_dir=args.room_dir, factor=args.factor, normalize=True, test_every=10**9)
    c2ws = np.asarray(parser.camtoworlds)

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device)
    sp = ck["splats"] if "splats" in ck else ck
    means = sp["means"].to(device)
    scales = torch.exp(sp["scales"]).to(device)
    quats = sp["quats"].to(device)
    op = sp["opacities"].to(device)
    opacities = torch.sigmoid(op.squeeze(-1) if op.dim() > 1 else op)
    colors = torch.cat([sp["sh0"], sp["shN"]], dim=1).to(device)

    frames = []
    with torch.no_grad():
        for j in range(0, len(c2ws), args.stride):
            cid = parser.camera_ids[j]
            K = torch.from_numpy(np.asarray(parser.Ks_dict[cid], np.float32)).to(device)[None]
            W, H = parser.imsize_dict[cid]
            W, H = int(W), int(H)
            viewmat = torch.from_numpy(np.linalg.inv(c2ws[j])).float().to(device)[None]
            out, _, _ = rasterization(
                means, quats, scales, opacities, colors, viewmat, K, W, H,
                sh_degree=args.sh_degree, render_mode="RGB",
                near_plane=0.01, far_plane=1e10, packed=True, rasterize_mode="classic")
            arr = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            if args.compare:
                real = np.asarray(Image.open(parser.image_paths[j]).convert("RGB").resize((W, H)), np.uint8)
                arr = np.hstack([real, arr])           # original | reconstruction
            h, w = arr.shape[:2]
            frames.append(arr[: h - (h % 2), : w - (w % 2)])   # even dims for h264
    imageio.mimwrite(args.out, frames, fps=args.fps, quality=8, macro_block_size=1)
    print(f"wrote {args.out} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
