# inhabit-kernel — from-scratch reconstruction kernel (experimental)

A kernel-level rebuild of the reconstruction stack: one unified, GPU-vectorised
PyTorch core instead of gluing separate reconstruction repos + Open3D TSDF + Poisson.
This is the experimental rebuild branch; `main` holds the public submission.

## What it is

| file | role |
|---|---|
| `kernel.py` | the unified GPU core: forward ray-sampling fusion (confidence-weighted, robust), free-space carving, object-aware voting, GPU edge-preserving denoise, and a from-scratch GPU **Surface Nets** mesher |
| `scene.py` | analytic synthetic room + objects, ray-cast depth, GT, visibility cull (no external renderer) |
| `bench.py` | noise-sweep benchmark vs Open3D TSDF + per-object decomposition / physics-readiness report |
| `front_end.py` | monocular metric depth front end (Depth-Anything-V2) |
| `vggt_front.py` | multi-view feed-forward front end (VGGT-1B) |
| `bench_real.py` / `bench_vggt.py` | real-data benchmarks on Replica: GT vs predicted depth through the kernel |
| `export_assets.py` / `eval_phys.py` | export separable physics-ready assets + domain-randomized Genesis physics eval |
| `RESULTS.md` | all benchmark numbers and honest findings |

## Headline results (see RESULTS.md for the full tables)

- **Two surface modes** (`kernel.py` TSDF for speed, `surfel.py` for precision). Net:
  the rebuild **matches Open3D on clean data and beats it under noise**, synthetic and
  real. On real Replica room0 the surfel reaches Chamfer 1.25 / F@2cm 0.917 vs Open3D
  1.27 / 0.912; at 5 % synthetic noise the surfel+refine beats Open3D on every metric.
- **Fusion is 0.04 s and noise-invariant** (7–37× faster than Open3D's fusion); with
  the GPU Surface Nets mesher, **total TSDF time beats Open3D by 1.6–7.8×**.
- **Separable, physics-ready objects** dropped inside the reconstructed room (CoACD
  collider): 8/8 no tunnelling under domain randomization.
- **Feed-forward front ends on real RGB**: multi-view VGGT (3.80 cm) ≫ monocular
  (7.78 cm); fully pose-free runs but is pose-limited.

New files since the first commit: `surfel.py` (the precision representation, #12),
`complete.py` (amodal completion), `bench_surfel.py` / `bench_complete.py` /
`bench_vggt_pf.py` (benchmarks).
- **Separable per-object meshes** (sphere/box 0.7–1.0 cm), made physics-ready
  (watertight collision proxies + mass), passing a domain-randomized Genesis drop
  eval (12/12 no tunnelling).
- **Feed-forward front end on real RGB** (Replica room0): multi-view VGGT reaches
  4.38 cm vs GT-depth 2.15 cm and monocular 7.78 cm — multi-view ≫ monocular.

## Running

CPU stages need `skimage`, `trimesh`, `scipy`, `open3d`; the front ends need
`transformers==4.45.2` (Depth-Anything-V2) and the `vggt` package. Physics needs
the Genesis env (`numpy<2`). Benchmarks were run on an RTX 4070 Ti SUPER.

```bash
python bench.py --device cuda --W 256 --H 192 --ncam 48 --voxel 0.02   # fusion sweep
python bench_vggt.py --scene room0 --stride 70 --voxel 0.03             # feed-forward front end
```
