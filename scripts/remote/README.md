# scripts/remote — the GPU reconstruction + benchmark backend

These are the exact, runnable drivers for the heavy stages that can't run on a
laptop: training the three reconstructors and scoring them against ground-truth
meshes. Each method ships its own CUDA/conda environment; we drive them over SSH
to a GPU box rather than pretend a single `pip install` reproduces 6 GPU-hours.

## Helper

- **`tb.sh`** — runs a command on the GPU box over a 2-hop SSH proxy, piping into
  `bash -s` (the remote login shell is csh). `./tb.sh '<remote bash>'`.

## Per-method setup + train (run once each)

| Method | Setup | Train → mesh |
|---|---|---|
| PGSR | `setup_pgsr.sh` | `launch_pgsr.sh <scene>` → `render.py` TSDF mesh |
| DN-Splatter | `setup_dnsplat.sh` | `ns-train dn-splatter …` → `gs-mesh o3dtsdf` |
| MonoSDF | `setup_monosdf.sh` | `launch_monosdf_tight.sh` → `extract_monosdf_mesh.sh` |

`setup_*.sh` clone the upstream repos, build CUDA extensions (sm_89), and pin the
working torch/gsplat wheels. `room_grids_tight.conf` is the MonoSDF config tuned
for a 16 GB card.

## Fusion + benchmark

- **`fuse_consensus.py`** — the gated consensus fusion (also packaged at
  `src/vid2scene/fuse/consensus.py` for laptop use). `--identity-dn` when both
  meshes are already in the same (COLMAP/metric) frame.
- **`run_bench_parallel.sh`** — memory-safe 2-stream multi-scene orchestrator
  (one PGSR + one DN stream concurrent on a single 16 GB GPU).
- **`run_dn_backbone.sh`** — re-fuse with DN-Splatter as backbone (the ablation).
- Eval uses DN-Splatter's `eval_mesh_vis_cull.py` (visibility-culls GT + pred,
  then Chamfer / F-score). Collate the resulting JSONs with `vid2scene benchmark`.

## Diagnostics / viz helpers

`render_interior.py`, `render_aligned.py` (render meshes from shared poses),
`cam_coverage.py` (top-down capture-coverage plot), `decimate_meshes.py`,
`process_monosdf_tight.py`.

See [`../../docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) and
[`../../docs/BENCHMARK.md`](../../docs/BENCHMARK.md).
