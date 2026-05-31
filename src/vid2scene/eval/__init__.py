"""Stage 5 — evaluation & sanity checks (PLANNED).

With ground truth (ScanNet++):  Chamfer distance, F-score@5cm; pose ATE/RPE
   via `evo`; novel-view PSNR/SSIM/LPIPS on held-out frames.
Without ground truth (own room): held-out-view PSNR, scale check vs a measured
   object, plane-fit RMS for floor/walls, MapAnything confidence coverage %.

The headline table is pose-free vs pose-assisted across these metrics.
"""
