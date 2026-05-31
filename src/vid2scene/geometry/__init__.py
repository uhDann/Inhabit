"""Stage 2 — feed-forward geometry (PLANNED, GPU/RunPod).

Backbone: MapAnything (Meta, Apache code + Apache weights variant, metric,
accepts optional poses). VGGT is the runner-up.

Two runs share this module, which is the project's originality hook:
  * pose-free            — frames only
  * pose-assisted        — frames + ARKit poses/intrinsics/timestamps

Planned interface:
    def reconstruct(frames_dir, out_dir, poses: Path | None = None) -> ReconResult
returning a metric point cloud, per-view depth, camera poses, and confidence.
"""
