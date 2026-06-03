# Object separation fix (Tier 0 + Tier 1)

The old separation was weak because per-frame 2D masks were inconsistent across views: the
same chair got a different id in every frame, so per-voxel votes smeared. This package
fixes the **inputs** to the voting the kernel already does, not the voting itself.

What changed, and why:

1. **View-consistent masks (`seg.py`).** SAM 2 seeds objects on frame 0 (automatic mask
   generator) and PROPAGATES them through the video, so each object keeps one stable id
   across all frames. That is the single biggest fix: votes now agree across views.
2. **Hungarian re-association (`seg.py`).** If you segment a long capture in chunks, merge
   the id spaces by max-IoU assignment so ids stay global.
3. **Superpoint pooling (`superpoints.py`).** Votes are pooled over normal-coherent mesh
   superpoints instead of raw voxels. A superpoint never crosses a chair-leg/floor normal
   discontinuity, so contact patches separate cleanly and one bad mask can't split an
   object. A superpoint commits to an object only when its top vote beats the runner-up by
   a margin; otherwise it falls back to the room shell. This kills flicker and bleed.

## Run

```
python -m separation.run \
  --frames_dir   frames/ \            # temporally ordered jpgs for SAM2
  --depth_glob  'depth/*.npy' \       # posed depth aligned to frames
  --poses        poses.npy \          # [N,4,4] cam->world
  --K            600 600 599.5 339.5 \
  --bounds      -3 -3 -1  3 3 3 \     # scene AABB (meters)
  --voxel        0.02 \
  --sam2_cfg     sam2_hiera_l.yaml --sam2_ckpt sam2_hiera_large.pt \
  --out          runs/separation
```

Output: `room_shell.ply` + `object_NN.ply` per separated object, ready for the photoreal
appearance pass and for CoACD/Genesis physics export.

## First-run caveats
- SAM 2 API names shift between releases. `add_new_points_or_box`, `init_state`,
  `propagate_in_video` are the current (2.1) names; adapt if your checkout differs.
- `init_state(video_path=...)` expects a dir of frames; we pass `frames_dir` directly.
- Depth loader assumes Replica's 6553.5 PNG scale or raw float `.npy`; adapt `load_depth`.
- The kernel must be built with `n_labels >= number of tracked objects` (the driver
  sets this automatically from the masks).
