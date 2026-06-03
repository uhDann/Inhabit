# Next-gen: photoreal mesh + clean object separation

Two upgrades that sit on top of the existing kernel (metric surfel geometry, ~1 cm). Both
are fully written and ready to run; geometry is the frozen asset, these add appearance and
separation. All code is staged so the moment a GPU is free we run end to end with no
further writing.

```
phone video / posed depth
        |
   [ kernel ]  ---->  metric mesh (frozen geometry, our ~1cm asset)
        |                         |
        |                         +--> [ separation/ ]  room shell + per-object meshes
        |                         |        SAM2 video masks -> kernel votes -> superpoints
        |                         |
        +-----------------------> [ photoreal/ ]  splat-quality appearance ON the mesh
                                           per-texel diffuse + SG specular, nvdiffrast
                                           L1+LPIPS fine-tune, splat-appearance distill
                                           held-out NVS + 2AFC confusion study
```

## Run order (GPU)

0. **Deps:** `pip install -r requirements_nextgen.txt` and install SAM 2 + a 3DGS trainer
   (gsplat) per their repos. Download SAM 2 + Depth-Anything/VGGT checkpoints.

1. **Separation** (independent of photoreal, run in parallel):
   ```
   python -m separation.run --frames_dir ... --depth_glob ... --poses ... \
       --K fx fy cx cy --bounds xmin ymin zmin xmax ymax zmax \
       --sam2_cfg ... --sam2_ckpt ... --out runs/separation
   ```
   Gives `room_shell.ply` + `object_NN.ply`.

2. **Photoreal appearance** on the frozen mesh (see `photoreal/README.md`):
   ```
   python -m photoreal.train  --mesh <mesh.ply> --data <scene> --out runs/pr   # bake appearance
   python -m photoreal.eval   --run runs/pr --data <scene>                     # held-out NVS metrics
   python -m photoreal.distill --mesh <mesh.ply> --data <scene> --splat_dir <3dgs> --out runs/prd  # novel piece
   python -m photoreal.eval   --run runs/prd --data <scene> --splat_dir <3dgs>  # mesh-vs-splat-vs-real
   python -m photoreal.twoafc --eval runs/prd --out runs/prd/study.html         # confusion study
   ```

3. **Targets** (held-out, every-8th-frame split): LPIPS <= 0.05, SSIM >= 0.95,
   PSNR >= 30, mesh within ~0.02 LPIPS of the tuned 3DGS, 2AFC discrimination ~50%.

## What is novel here
- **Splat-appearance distillation onto an accurate mesh** (`photoreal/distill.py`): a 3DGS
  is used only as an appearance teacher; its view-dependent radiance from arbitrary
  synthesised poses is distilled into the mesh's per-texel SG/neural appearance. The mesh
  inherits the splat look while staying true, editable, physics-ready geometry.
- **Confusion benchmark** (`photoreal/eval.py` + `twoafc.py`): paired mesh-vs-splat-vs-real
  held-out NVS plus a 2AFC human study on phone-captured rooms. No clean public benchmark
  proves "photoreal mesh confusable with the real room."
- **Superpoint-pooled, view-consistent separation** (`separation/`): fixes the smearing
  that flat per-frame masks caused, so contact patches separate cleanly.

See `PHOTOREAL_MESH_PLAN.md` (experiments E1-E7, hypotheses H1-H6) and
`RESEARCH_ROADMAP.md` (separation tiers, crack-level photometric path) for the full
research synthesis and the honest framing of what is a credible path vs an open frontier.

## Status
All modules written for correctness-by-construction; untested end to end (no GPU yet).
First-run caveats are documented in each package README (camera convention check, gsplat
ply field names, SAM 2 API version, depth scale). Run order above is the validation plan.
