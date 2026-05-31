"""Stage 2 — MapAnything reconstruction, pose-free or pose-assisted.

Same code path for both modes (controlled A/B): `--use-poses` feeds the phone's
ARKit camera poses + intrinsics into MapAnything's multi-modal `infer`; without
it, the model runs image-only and infers everything itself.

ARKit gives a column-major 4x4 cam->world transform in the OpenGL/RUB camera
convention (+X right, +Y up, -Z forward). MapAnything wants OpenCV cam->world
(+X right, +Y down, +Z forward), so we transpose the flattened matrix and flip
the camera Y/Z axes (diag(1,-1,-1)). ARKit poses are metric, so is_metric_scale
is True and the reconstruction comes out in real-world units.

Run on the GPU box (needs the mapanything env):
    python -m vid2scene.geometry.reconstruct \
        --frames-dir runs/room0_ingest/frames \
        --report     runs/room0_ingest/frame_report.json \
        --poses      data/room0/frames.json \
        --output     runs/room0_ingest/recon_pose_assisted.glb \
        --apache --use-poses
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def arkit_pose_to_opencv_c2w(camera_pose_16: list[float]) -> np.ndarray:
    """ARKit flat column-major cam2world (RUB) -> OpenCV cam2world (RDF) 4x4."""
    m = np.array(camera_pose_16, dtype=np.float64).reshape(4, 4).T  # column-major -> (row,col)
    out = m.copy()
    out[:3, :3] = m[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # flip camera Y and Z
    return out.astype(np.float32)


def keyframe_to_source_index(report_json: str) -> list[int]:
    """frame_{k:04d}.jpg corresponds to the k-th smallest selected source index."""
    rep = json.load(open(report_json))
    return sorted(f["index"] for f in rep["frames"] if f["selected"])


def build_views(frames_dir, report_json, poses_json, use_poses):
    import torch
    from PIL import Image

    sources = keyframe_to_source_index(report_json)
    pdata = json.load(open(poses_json))
    fx, fy, cx, cy = pdata["intrinsics"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    frames_dir = Path(frames_dir)

    views = []
    for k, src in enumerate(sources):
        p = frames_dir / f"frame_{k:04d}.jpg"
        if not p.exists():
            continue
        img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)  # (H, W, 3)
        view = {"img": torch.from_numpy(img)}
        if use_poses:
            view["intrinsics"] = torch.from_numpy(K)
            pose = arkit_pose_to_opencv_c2w(pdata["frames"][src]["cameraPose"])
            view["camera_poses"] = torch.from_numpy(pose)
            view["is_metric_scale"] = torch.tensor([True])
        views.append(view)
    return views


def export_glb(outputs, output_path, as_mesh):
    import torch  # noqa: F401
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.viz import predictions_to_glb

    world_points, images, masks = [], [], []
    for pred in outputs:
        depth = pred["depth_z"][0].squeeze(-1)
        intr = pred["intrinsics"][0]
        pose = pred["camera_poses"][0]
        pts3d, valid = depthmap_to_world_frame(depth, intr, pose)
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool) & valid.cpu().numpy()
        world_points.append(pts3d.cpu().numpy())
        images.append(pred["img_no_norm"][0].cpu().numpy())
        masks.append(mask)

    preds = {
        "world_points": np.stack(world_points),
        "images": np.stack(images),
        "final_masks": np.stack(masks),
    }
    scene = predictions_to_glb(preds, as_mesh=as_mesh)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)


def _write_ply_bin(path, pts, cols):
    """Write a binary little-endian colored point-cloud PLY."""
    n = len(pts)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        rec.tofile(f)


def write_ply(outputs, path, max_points=2_000_000, crop_quantile=0.99):
    """Colored point cloud straight from model outputs, with flyer removal + subsample."""
    from mapanything.utils.geometry import depthmap_to_world_frame

    pts_all, col_all = [], []
    for pred in outputs:
        depth = pred["depth_z"][0].squeeze(-1)
        intr = pred["intrinsics"][0]
        pose = pred["camera_poses"][0]
        pts3d, valid = depthmap_to_world_frame(depth, intr, pose)
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool) & valid.cpu().numpy()
        pts_all.append(pts3d.cpu().numpy()[mask])
        col_all.append(pred["img_no_norm"][0].cpu().numpy()[mask])
    pts = np.concatenate(pts_all)
    cols = np.concatenate(col_all)
    cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8) if cols.max() <= 1.0 else cols.astype(np.uint8)

    # drop flyers far from the scene centroid
    center = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    keep = dist < np.quantile(dist, crop_quantile)
    pts, cols = pts[keep], cols[keep]

    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _write_ply_bin(path, pts, cols[:, :3])
    print(f"wrote {path}  ({len(pts)} points)")


def main():
    ap = argparse.ArgumentParser(description="MapAnything pose-free / pose-assisted reconstruction")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--report", required=True, help="frame_report.json from ingest")
    ap.add_argument("--poses", required=True, help="frames.json (ARKit poses/intrinsics)")
    ap.add_argument("--output", required=True, help="output .glb")
    ap.add_argument("--use-poses", action="store_true", help="feed ARKit poses+intrinsics")
    ap.add_argument("--apache", action="store_true", help="use Apache-licensed weights")
    ap.add_argument("--point-cloud", action="store_true", help="export points instead of mesh")
    ap.add_argument("--ply", help="also write a downsampled colored .ply for web viewing")
    ap.add_argument("--max-points", type=int, default=2_000_000, help="max points in --ply output")
    ap.add_argument("--confidence-percentile", type=int, default=0,
                    help="drop this bottom %% of low-confidence pixels (0 = off)")
    ap.add_argument("--multiview-confidence", action="store_true",
                    help="use multi-view depth-consistency confidence (removes floaters)")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.device import get_device
    from mapanything.utils.image import preprocess_inputs

    device = get_device()
    name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    print(f"device={device}  model={name}  use_poses={args.use_poses}")
    model = MapAnything.from_pretrained(name).to(device)

    views = build_views(args.frames_dir, args.report, args.poses, args.use_poses)
    print(f"built {len(views)} views")
    processed = preprocess_inputs(views)

    outputs = model.infer(
        processed,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=args.confidence_percentile > 0,
        confidence_percentile=args.confidence_percentile,
        use_multiview_confidence=args.multiview_confidence,
        ignore_pose_inputs=not args.use_poses,
        ignore_calibration_inputs=not args.use_poses,
    )
    print("inference complete")

    export_glb(outputs, args.output, as_mesh=not args.point_cloud)
    print(f"saved {args.output}")

    if args.ply:
        write_ply(outputs, args.ply, max_points=args.max_points)


if __name__ == "__main__":
    main()
