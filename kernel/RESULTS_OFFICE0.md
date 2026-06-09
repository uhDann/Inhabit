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

## Generalization (3 Replica scenes, same pipeline)

| Scene | Splat PSNR | Textured-mesh PSNR | SSIM | Instances |
|---|---|---|---|---|
| office0 | 43.2 | 33.1 | 0.969 | 23 |
| room0 | 35.4 | 28.4 | 0.880 | 67 |
| office1 | 36.1 | 29.9 | 0.904 | 21 |

## Closing the geometry gap (full-res, 63 held-out views)

The half-res PSNR (33) was partly a downsampling artifact; at native 1200x680 the gap is
real. Differentiable vertex refinement (`photoreal/refine_vertices.py`) closes a meaningful
chunk of it -- but ONLY with strong Laplacian smoothing:

| office0 (full res) | PSNR | SSIM | LPIPS | FID(real) |
|---|---|---|---|---|
| mesh-from-splat (baseline) | 26.7 | 0.937 | 0.234 | 74.3 |
| + vertex refine, weak reg (LAP 8) | 34.1 | 0.925 | 0.339 | **98.0** (worse) |
| + vertex refine, **strong reg (LAP 40)** | **35.6** | **0.944** | 0.288 | **61.2** |
| splat (reference) | 39.0 | 0.966 | 0.232 | 30.7 |

Non-obvious finding: the method is **regularisation-gated**. Weak smoothness overfits train
views with rough deformation and makes FID WORSE; strong smoothness gives the alignment gain
(PSNR +8.9 dB, SSIM up, FID 74->61) without the realism penalty. Mean vertex offset 48mm
(weak) vs 27mm (strong). A naive "just optimise vertices against photometry" reimplementation
is actively harmful. Confidence-masked depth fusion was tried first and FAILED (FID 74->167).

## Task-transfer: measuring the reconstruction by ROBOT UTILITY (`eval_navmesh.py`)

The field benchmarks reconstructions on pixels (PSNR/FID), which it admits do not predict
downstream task success. We measure NAVIGABILITY transfer instead: recompute a robot navmesh
(Habitat/Recast) on the GT scene and on our reconstructed twin, then check how many real
shortest paths reproduce in the twin.

Across 3 scenes (twin = PGSR mesh), navigation paths reproducible in the twin (0% median
path-length error unless noted):

| scene | nav paths reproducible | path-len err | physics-completion stable |
|---|---|---|---|
| office0 | 81/100 | 0% | 82% |
| room0 | 66/100 | 0% | 88% |
| office1 | 82/100 | 3% | 33% |

**Key finding -- task-driven fidelity:** office1 is the WORST scene for physics (33% stable)
yet the BEST for navigation (82% paths). Navigation only needs the floor + large obstacles
right (a rough reconstruction nails that); physics stability needs every object solid (a
rough reconstruction fails). So DIFFERENT robot tasks require DIFFERENT reconstruction
fidelity -- the under-measured axis the field flags, shown here with numbers. This is the
differentiated, field-relevant contribution (task utility, not pixels).

## Surface-aligned mesh (PGSR) replaces the lossy TSDF handoff

Our biggest geometry weakness was extracting the mesh from the splat's *expected* depth
(edge bias, foreground-only). Training PGSR (surface-aligned splatting; `replica_to_colmap.py`
prepares the data) and fusing its unbiased depth gives a cleaner mesh:

| office0 (full res) | PSNR | SSIM | LPIPS | FID |
|---|---|---|---|---|
| our mesh-from-splat (TSDF) | 26.7 | 0.937 | 0.234 | 74.3 |
| **PGSR mesh (surface-aligned)** | 26.9 | 0.945 | **0.203** | **61.4** |
| splat | 39.0 | 0.966 | 0.232 | 30.7 |

PGSR matches our vertex-refinement result (FID ~61) for free and beats the splat on LPIPS.
PSNR similar (the ~12 dB mesh-vs-splat gap is alignment, unchanged). The real payoff is
downstream: cleaner geometry -> crisper separation -> physics-completion -> 82% stable / 2.9 cm
mean drift (see `physics/README.md`).

## Experiments that did NOT help (honest)

- **Splat->mesh distillation** (`distill` style: bake + view-dep residual trained on real
  frames AND novel splat-rendered poses): **27.1 PSNR**, *worse* than the 33.1 projective
  bake. Novel-pose splat supervision on mismatched mesh geometry injects error. Bake wins.
- **Physics drop test** (`physics/`): the Genesis sim is valid (no OOM after CoACD), but the
  separated furniture are surface fragments, so only ~3/23 stay within 5 cm — they fall to
  the floor rather than rest as furniture. Both separation and physics are bottlenecked by
  the same upstream limit: **furniture geometry completeness**, not the method.

## Pipeline (office0)

```
splat_teacher.py   --replica office0 --mesh viz/office0_dec.ply --colors viz/office0_textured.ply
mesh_from_splat.py --replica office0 --teacher runs/gs/gs_teacher.pt --out viz/mesh_from_splat.ply
projective_bake.py --replica office0 --mesh viz/mesh_from_splat.ply --view_dep --save_mesh viz/room_textured.ply
instances.py       --mesh viz/mesh_from_splat.ply --out runs/inst
```

Open issue: separation is still over-fragmented on incomplete furniture geometry (drops
small parts). Next: distil the splat into the mesh, and improve furniture completeness.
