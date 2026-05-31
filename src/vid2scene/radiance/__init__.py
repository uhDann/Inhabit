"""Stage 3 — radiance field / splatting (PLANNED, GPU/RunPod).

Hero:     Gaussian Splatting (gsplat / Splatfacto), or AnySplat feed-forward
          for a seconds-fast path. Export .ply -> SuperSplat web viewer.
Baseline: nerfacto (NeRF) trained from the *same* poses, for an honest
          radiance-field vs splatting comparison.

Planned interface:
    def splat(frames_dir, poses, out_dir, method="splatfacto") -> Path  # .ply/.splat
    def nerf(frames_dir, poses, out_dir) -> Path                        # baseline
"""
