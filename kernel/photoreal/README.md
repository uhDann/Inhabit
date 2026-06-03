# photoreal — splat-quality appearance on our true mesh

Goal: a real 3D mesh that renders as photorealistically as a Gaussian splat
("confusable with the real room"), while staying editable + physics-ready. The mesh
geometry comes from our kernel (the asset); here we learn only **appearance**.

## Pieces
| file | role |
|---|---|
| `core.py` | cameras (OpenCV→OpenGL), UV unwrap (xatlas), `DeferredAppearance` (per-texel diffuse + feature atlas + tiny view-dependent MLP), nvdiffrast `Renderer` with anti-aliasing |
| `data.py` | posed-frame dataset (Replica or generic folder); held-out every-8th-frame split |
| `train.py` | optimise appearance on a FIXED mesh under L1 + LPIPS vs real frames |
| `distill.py` | **novel**: distil a trained 3DGS's appearance onto our mesh (real-frame + novel-pose supervision) |
| `eval.py` | held-out NVS metrics: PSNR / SSIM / LPIPS / DreamSim, side-by-side panels |
| `twoafc.py` | builds a 2AFC "which is the real photo?" HTML study from the eval panels |

## Dependencies (GPU box)
```
pip install nvdiffrast xatlas lpips dreamsim plyfile scikit-image trimesh imageio
# nvdiffrast needs a CUDA toolchain; gsplat needed only for distill.py:
pip install gsplat
```

## Run order (when the GPU is available)
```bash
# 0. (verify camera convention once)
python -c "from photoreal.core import core_selftest; ..."   # render bare mesh from a known pose

# 1. baseline appearance from the real frames
python -m photoreal.train  --mesh office0_surfel.ply --replica <Replica>/office0 --out runs/pr

# 2. (optional, the novel path) train a gsplat teacher with the existing gsplat pipeline,
#    then distil it onto the mesh:
python -m photoreal.distill --mesh office0_surfel.ply --teacher splat.ply --replica <Replica>/office0 --out runs/pr_distill

# 3. evaluate on held-out views (add --splat_dir to compare against the splat)
python -m photoreal.eval --mesh office0_surfel.ply --appearance runs/pr/appearance.pt --replica <Replica>/office0 --out runs/pr/eval

# 4. build the human 2AFC study and open it
python -m photoreal.twoafc --eval_dir runs/pr/eval --out runs/pr/twoafc.html
```

## Targets ("indistinguishable from real")
LPIPS ≤ 0.05, SSIM ≥ 0.95, PSNR ≥ 30 dB on held-out views; within ~0.02 LPIPS of a
tuned 3DGS on the same capture; **2AFC discrimination ≈ 50% (chance)**.

## First-run caveats (untested — written without a GPU)
- Verify the OpenCV→OpenGL camera convention with `core_selftest` (a mis-set sign shows
  as a flipped/empty render); adjust `_CV2GL` / `opengl_projection` if needed.
- `_load_gaussian_ply` in `distill.py` assumes a standard 3DGS ply (f_dc SH0 + scale/rot);
  adapt field names to your gsplat export. gsplat `rasterization` arg order may need a
  tweak per your gsplat version.
- The MLP residual is zero-initialised so training starts from pure diffuse and adds
  view-dependence — expect L1 to drop fast, LPIPS slower.
