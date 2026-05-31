# Architecture & repo map

vid2scene turns a phone video of a room into a **metric** 3D reconstruction, makes
the geometry **trustworthy** (multiple methods + consensus fusion + a GT-mesh
benchmark), and turns the result into a **robot-explorable world**.

## Pipeline

```
phone video
   │
   ▼
[1] ingest     frame selection (blur gate + parallax spacing)            CPU ·  src/vid2scene/ingest
   │           → keyframes + report.html
   ▼
[2] reconstruct  THREE methods on the same frames, on a CUDA box:        GPU ·  scripts/remote
   │             • PGSR        (planar 3DGS → TSDF mesh)
   │             • DN-Splatter (3DGS + mono depth/normal priors → mesh)
   │             • MonoSDF     (neural SDF + mono priors → marching cubes)
   │           → one coloured mesh per method (+ a reference Gaussian splat)
   ▼
[3] fuse       consensus gap-fill: trusted backbone + donor-where-holes  CPU ·  src/vid2scene/fuse
   │           → a single fused mesh (the original contribution)
   ▼
[4] benchmark  score every mesh vs Replica GT (Chamfer / F-score),       GPU cull + CPU collate
   │           visibility-culled                                                src/vid2scene/benchmark
   │           → the comparison table (docs/BENCHMARK.md)
   ▼
[5] embodied   metric mesh → gravity-aligned sim collider; robot         CPU export · GPU sim
               navigates, camera re-rendered from the splat              src/vid2scene/embodied
               → Habitat walkthrough today; Genesis RL env (proposed)           scripts/habitat_*
```

## Where things live

| Path | What |
|---|---|
| `src/vid2scene/ingest/` | **[CPU]** frame selection + HTML validation report |
| `src/vid2scene/fuse/` | **[CPU]** consensus gap-fill fusion (`consensus.py`) — the novel bit |
| `src/vid2scene/benchmark/` | **[CPU]** collate GT-mesh metrics → Markdown table |
| `src/vid2scene/viz/` | **[CPU]** mesh decimation for the web viewers |
| `src/vid2scene/embodied/` | **[CPU]** export sim-ready GLB (Genesis/Habitat hand-off) |
| `src/vid2scene/geometry/`, `radiance/` | early MapAnything/gsplat scaffolding (superseded by the remote method drivers; kept for reference) |
| `scripts/remote/` | **[GPU]** the reconstruction + benchmark backend: per-method `setup_*.sh`/`launch_*.sh`, `fuse_consensus.py`, `eval` orchestration, `tb.sh` (2-hop SSH helper), `run_bench_parallel.sh` |
| `scripts/` | local helpers: TSDF mesh, Habitat navmesh + path recording, photoreal path render, pruning |
| `viewer/` | three.js / GaussianSplats3D web viewers |
| `runs/replica_eval/` | the bundled GT-mesh metric JSONs (so `vid2scene benchmark` reproduces the table offline) |
| `docs/` | `BENCHMARK.md` (protocol + results), `PHASE2_GENESIS.md` (embodied upgrade), `FINDINGS.md` (engineering log) |

## Why two execution environments

The CPU stages are a normal `pip install -e .` package and run on a laptop —
this is what makes the pipeline **usable** (fuse / benchmark / decimate / export
any meshes you have). The reconstruction stage runs three separate SOTA repos
(PGSR, DN-Splatter, MonoSDF), each with its own CUDA/conda environment and
multi-hour training; those are driven by `scripts/remote/` on a GPU box. We keep
them separate rather than pretend a single `pip install` reproduces 6 GPU-hours
of training — `scripts/remote/*.sh` are the exact, runnable drivers.

## Run it

```bash
make install                 # CPU stages
make benchmark               # reproduce the GT-mesh table from bundled results
make viewer                  # serve viewer/ at :8765, then open the pages
vid2scene fuse --backbone pgsr.ply --donor dn.ply --out consensus.ply
vid2scene embodied --mesh consensus.ply --out room_sim.glb --scene-json scene.json
```
