# Photoreal mesh + object separation — office0 results

Held-out novel-view synthesis on Replica office0 (every-8th-frame split, scale 0.5).

| Reconstruction | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 3DGS splat teacher (`photoreal/splat_teacher.py`) | **43.2** | **0.987** | **0.066** |
| Textured mesh, projective bake + view-dep (`photoreal/projective_bake.py`) | **33.1** | 0.969 | 0.120 |
| Textured mesh, bake only | 31.9 | 0.953 | 0.145 |
| GT-RGBD reference mesh | 26.3 | — | — |

## The bug that mattered: nvdiffrast vertical flip

Every mesh — including a gold-standard GT-RGBD reconstruction — scored ~15 dB PSNR; only
the splat (its own correct projection) hit 43. The uniformity was the tell: it was not
geometry. nvdiffrast's rasteriser uses a **bottom-left origin**, so mesh renders were
vertically flipped vs the top-left-origin GT image arrays, capping PSNR ~15 even on a
perfect mesh. Fix: negate the y-row of the OpenGL projection in `core.opengl_projection`
(`P[1,1] = -2fy/H`, `P[1,2] = 1 - 2cy/H`). This lifted every mesh ~11 dB and also fixed the
bake's depth-occlusion buffer. Diagnostic rule: if a perfect GT-RGBD mesh renders to ~15 dB,
the eval/render is broken, not the geometry.

## What works (and what didn't)

- **Texture:** deterministic projective per-vertex bake (depth-occluded, cos^4/z^2 view
  weighting, graph hole-fill) beats appearance optimisation (which collapses to a muddy mean
  colour). UV-atlas texture optimisation kept tripping over nvdiffrast texture conventions;
  per-vertex via `dr.interpolate` is the robust path. A small per-vertex view-dependent
  residual (feat + tiny MLP, L1 + Laplacian, +0.3·tanh) adds ~1 dB.
- **Separation:** SAM2-vote + superpoint pooling and SAI3D-style affinity merging both
  mislabel the floor (swallow furniture into the shell, or split the floor into objects).
  The robust recipe (`separation/instances.py`): RANSAC-remove only the **boundary**
  structural planes (floor/ceiling/walls at the scene bbox), keep interior furniture, then
  take **mesh connected components** = whole instances. Quality tracks geometry completeness
  — run it on the splat-derived mesh (`photoreal/mesh_from_splat.py`), not a thin decimated
  reconstruction.

## Pipeline (office0)

```
splat_teacher.py   --replica office0 --mesh viz/office0_dec.ply --colors viz/office0_textured.ply
mesh_from_splat.py --replica office0 --teacher runs/gs/gs_teacher.pt --out viz/mesh_from_splat.ply
projective_bake.py --replica office0 --mesh viz/mesh_from_splat.ply --view_dep --save_mesh viz/room_textured.ply
instances.py       --mesh viz/mesh_from_splat.ply --out runs/inst
```

Open issue: separation is still over-fragmented on incomplete furniture geometry (drops
small parts). Next: distil the splat into the mesh, and improve furniture completeness.
