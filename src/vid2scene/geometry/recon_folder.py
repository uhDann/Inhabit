"""Pose-free MapAnything on a plain folder of images (e.g. Mip-NeRF 360).

Generalizes the pipeline beyond our ARKit video: runs images-only inference,
exports a colored PLY (splat init) and a cameras.json containing, per view, the
MapAnything-estimated intrinsics + cam2world pose AND the processed-resolution
RGB image (so the splat trainer's cameras and images are mutually consistent).

    python -m vid2scene.geometry.recon_folder --image-dir <dir> \
        --out-ply out.ply --cameras-json cams.json --cams-img-dir cams_imgs --apache
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-ply", required=True)
    ap.add_argument("--cameras-json", required=True)
    ap.add_argument("--cams-img-dir", required=True, help="where processed-res training images are written (fallback)")
    ap.add_argument("--train-image-dir", help="use full-res images here (matched by filename) for splat training instead of MapAnything's ~518px output; intrinsics are rescaled accordingly")
    ap.add_argument("--apache", action="store_true")
    ap.add_argument("--max-views", type=int, default=80)
    ap.add_argument("--confidence-percentile", type=int, default=20)
    ap.add_argument("--multiview-confidence", action="store_true")
    ap.add_argument("--max-points", type=int, default=1_500_000)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from PIL import Image
    from mapanything.models import MapAnything
    from mapanything.utils.device import get_device
    from mapanything.utils.image import preprocess_inputs
    from vid2scene.geometry.reconstruct import write_ply

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = sorted(p for p in Path(args.image_dir).iterdir() if p.suffix in exts)
    if len(paths) > args.max_views:
        idx = np.linspace(0, len(paths) - 1, args.max_views).astype(int)
        paths = [paths[i] for i in idx]
    print(f"{len(paths)} input images", flush=True)

    device = get_device()
    name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    print(f"device={device} model={name}", flush=True)
    model = MapAnything.from_pretrained(name).to(device)

    views = [{"img": torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.uint8))} for p in paths]
    processed = preprocess_inputs(views)
    outputs = model.infer(
        processed, memory_efficient_inference=True, minibatch_size=1, use_amp=True, amp_dtype="bf16",
        apply_mask=True, mask_edges=True,
        apply_confidence_mask=args.confidence_percentile > 0,
        confidence_percentile=args.confidence_percentile,
        use_multiview_confidence=args.multiview_confidence,
        ignore_calibration_inputs=True, ignore_pose_inputs=True,
    )
    print("inference done", flush=True)

    write_ply(outputs, args.out_ply, max_points=args.max_points)

    cdir = Path(args.cams_img_dir); cdir.mkdir(parents=True, exist_ok=True)
    cams = []
    for i, pred in enumerate(outputs):
        K = pred["intrinsics"][0].cpu().numpy().astype(np.float64).copy()  # at processed (~518px) res
        proc_h, proc_w = pred["img_no_norm"][0].shape[:2]
        img_path = None
        if args.train_image_dir:
            tp = Path(args.train_image_dir) / paths[i].name
            if tp.exists():
                tw, th = Image.open(tp).size
                sx, sy = tw / proc_w, th / proc_h            # rescale intrinsics to full-res
                K[0, 0] *= sx; K[0, 2] *= sx; K[1, 1] *= sy; K[1, 2] *= sy
                img_path = str(tp)
        if img_path is None:                                  # fallback: MapAnything's processed image
            fn = cdir / f"view_{i:04d}.png"
            Image.fromarray((pred["img_no_norm"][0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).save(fn)
            img_path = str(fn)
        cams.append({
            "image": img_path,
            "K": K.tolist(),
            "c2w": pred["camera_poses"][0].cpu().numpy().tolist(),  # OpenCV cam2world
        })
    Path(args.cameras_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"views": cams}, open(args.cameras_json, "w"))
    print(f"wrote {args.cameras_json} ({len(cams)} views) + {args.out_ply}", flush=True)


if __name__ == "__main__":
    main()
