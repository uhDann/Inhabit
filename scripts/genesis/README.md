# Phase 2 — Genesis embodied stage (REAL, on UCL trailbreaker GPU)

A robot in the scanned room under physics, running on Genesis (`genesis-world` 1.0.0,
Apache-2.0 physics core) on trailbreaker's RTX 4070 Ti SUPER (sm_89, Ada).

## Result summary

- **genesis-world install + `gs.init()` on GPU: YES.** Genesis 1.0.0 initializes on the
  `gs.cuda` backend (RTX 4070 Ti SUPER detected) and on `gs.cpu`. Offscreen EGL rendering
  (Rasterizer) also works.
- **Step 1 (sphere drop): PASS.** A rigid sphere (r=0.12 m) dropped above the real
  reconstructed Replica room0 floor falls under gravity, collides, and comes to rest at
  base z ≈ 0.098 m (= floor 0 + radius 0.12, within 2 cm) at a clear floor spot. No
  tunnelling, no fall to infinity. Validates mesh import + metric scale + collision + gravity.
- **Step 2 (Go2 quadruped): STAND = PASS.** Genesis-shipped Go2 URDF spawns at z=0.42 and
  settles to a stable standing base height of z ≈ 0.29 m on the room floor, held for 4 s.
  Forward locomotion: the hand-tuned open-loop trot keeps it balanced in place but produces
  no net travel (a trained RL policy would be needed for robust walking — out of scope here).

## Environment

- Conda env: `/cs/student/projects3/2023/dkozlov/conda-envs/genesis`
  (fresh python 3.10 + pip; NOT a clone of `splat` — cloning the 8.7 GB env over the
  network FS was too slow, so a lean fresh env was used instead).
- Packages: `genesis-world==1.0.0`, `torch==2.2.0+cu121`, `numpy==1.26.4` (pinned <2 to
  fix the torch-2.2/numpy-2.x binary ABI break that genesis pulls in by default).
- Driver 560.35.03 (supports CUDA <= 12.6). System `/usr/local/cuda-12.6` is runtime-only;
  no nvcc was needed — Genesis uses the `quadrants` (taichi-derived) JIT compiler bundled
  in the wheel, plus `gs_madrona` for rendering. No source build required.

### Install recipe (reproduce)

```bash
CONDA=/cs/student/ug/2023/dkozlov/miniconda3/bin/conda
ENVS=/cs/student/projects3/2023/dkozlov/conda-envs
export PIP_CACHE_DIR=/cs/student/projects3/2023/dkozlov/pip-cache TMPDIR=/cs/student/projects3/2023/dkozlov/tmp
$CONDA create -y -p $ENVS/genesis python=3.10 pip
$ENVS/genesis/bin/pip install genesis-world          # 1.0.0
$ENVS/genesis/bin/pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
$ENVS/genesis/bin/pip install "numpy<2"              # ABI fix for torch 2.2
```

## Scripts

- `prep_room_mesh.py` — load the DN-Splatter Replica room0 TSDF mesh
  (`/cs/.../work/out/dn_room0/mesh/Open3dTSDFfusion_mesh.ply`), RANSAC the floor plane,
  gravity-align (floor -> z=0, +Z up), export OBJ (full + 200k-tri decimated). Run with the
  `splat` env python (open3d 0.19). Output: `room0_aligned_decim.obj` (floor z=0,
  ceiling z≈2.77, footprint x[-0.88,6.88] y[-1.19,3.51]).
- `step1_sphere_drop.py` — Step 1. `--backend gpu|cpu`, logs height-vs-time CSV+PNG,
  prints PASS/FAIL. Clean floor spot: `--drop_x 5.0 --drop_y 1.0 --drop_z 0.6`.
- `step2_go2.py` — Step 2. Go2 stand (+ optional `--walk` trot). kp=30, kv=1.5,
  symmetric standing pose; logs base z and forward x.

## Outputs (`outputs/`)

- `step1_sphere_height_vs_time.png` — GPU sphere settles at floor+radius (the PASS).
- `step1_sphere_cpu_clutter_spot.png` — CPU run at a cluttered XY (settles on furniture
  at 0.56 m — correct behavior, just a different surface).
- `step2_go2_stand_trajectory.png` — Go2 base z holds ~0.29 m for 4 s (stable stand).
- `genesis_room_render.png` — EGL render of the reconstructed room0 loaded as a collider.
- `go2_in_room_render.png` — Go2 placed in the room.

## What worked / what's blocked / what it would take to finish

- WORKED: install on GPU with no nvcc and no source build; metric mesh import + collision +
  gravity; sphere rest test; Go2 load + stable stand; offscreen rendering.
- KNOWN LIMITATIONS: (1) Genesis watertightens the room mesh into a fused collision wrap, so
  the effective collision surface can sit slightly above the visual floor where the
  reconstruction has clutter — pick a clean floor patch for clean drops. (2) Open-loop Go2
  gait does not walk forward; (3) genesis-world 1.0.0 emits a `torch<2.8` warning (non-fatal
  on 2.2.0) — a newer torch could be installed if any 2.8-only API is later needed.
- TO FINISH WALKING: drop in a pretrained Go2 RL locomotion policy (Genesis ships training
  examples in its GitHub repo, not the wheel) or Genesis's `genesis-nyx` for photoreal
  rendering (needs CUDA 12.9 — skipped per plan).
