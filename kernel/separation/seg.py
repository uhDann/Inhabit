"""Real per-frame, multi-view-consistent instance masks for the kernel's vote-based
object separation (the fix: the mask SOURCE, not the voting kernel).

Tier 0 recipe (research-backed): seed objects on the first frame with SAM 2's automatic
mask generator, then PROPAGATE them through the video with SAM 2's tracker so each
object keeps a stable global id across frames. Output: a per-frame [H,W] int32 label map
(0 = background/unassigned, 1..K = object ids) that feeds straight into the kernel's
per-voxel voting.

Optional Hungarian re-association is provided for cases where you segment in chunks and
need to merge id spaces.

Requires (GPU): sam2 (facebookresearch/sam2), torch, numpy, Pillow.
"""
from __future__ import annotations
import glob, os
import numpy as np


def sam2_video_masks(frames_dir, cfg, ckpt, device="cuda", min_area_frac=0.002,
                     max_objects=40):
    """frames_dir: a dir of sequential jpg frames (named so sorted() is temporal).
    Returns list of [H,W] int32 label maps (one per frame), ids consistent across frames."""
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    frames = sorted(glob.glob(f"{frames_dir}/*.jpg"))
    H, W = np.asarray(Image.open(frames[0])).shape[:2]
    min_area = int(min_area_frac * H * W)

    # --- seed objects on frame 0 with the automatic mask generator ---
    sam = build_sam2(cfg, ckpt, device=device)
    amg = SAM2AutomaticMaskGenerator(sam, points_per_side=24, min_mask_region_area=min_area)
    seed = amg.generate(np.asarray(Image.open(frames[0]).convert("RGB")))
    seed = sorted(seed, key=lambda m: -m["area"])[:max_objects]

    # --- propagate through the video ---
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
    state = predictor.init_state(video_path=frames_dir)
    for oid, m in enumerate(seed, start=1):
        # seed each object by its bounding box on frame 0
        x, y, w, h = m["bbox"]
        box = np.array([x, y, x + w, y + h], np.float32)
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=oid, box=box)

    labels = [np.zeros((H, W), np.int32) for _ in frames]
    for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        lab = labels[fidx]
        # paint larger objects first so small ones win overlaps (drawn last)
        order = np.argsort([-(ml > 0).sum().item() for ml in mask_logits])
        for k in order:
            oid = int(obj_ids[k]); mk = (mask_logits[k, 0] > 0).cpu().numpy()
            lab[mk] = oid
    return labels


def hungarian_relabel(masks_a_ids, masks_b_ids, iou):
    """Merge two id spaces by maximum-IoU assignment (scipy). iou[i,j] = IoU(a_i,b_j).
    Returns a dict b_id -> a_id for matched pairs (unmatched b keep new ids)."""
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-iou)
    return {int(masks_b_ids[j]): int(masks_a_ids[i]) for i, j in zip(r, c) if iou[i, j] > 0.3}
