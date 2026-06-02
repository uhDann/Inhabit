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

- **Fusion is 0.04 s and noise-invariant** (7–37× faster than Open3D's fusion).
- With the GPU Surface Nets mesher, **total time beats Open3D by 1.6–7.8×** across
  noise levels; meshing is no longer the bottleneck.
- **Quality wins in the noisy regime** (5% depth noise: Chamfer 2.6–3.1 vs 4.4 cm,
  F@5cm ~0.9 vs 0.6). Clean low-noise fine precision still favours mature TSDF.
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
