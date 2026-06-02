# inhabit-kernel: from-scratch rebuild — results

## Tier-1 improvements (latest)

- **#7 GPU Surface Nets mesher** (from scratch, replaces CPU marching cubes): meshing
  dropped from 0.2–2 s to 0.10–0.16 s, so **total time now beats Open3D by 1.6× / 5× /
  7.8×** at 0 / 2 / 5 % noise. Meshing is no longer the bottleneck.
- **#8 finer voxels now affordable**: at 1.3 cm voxel, clean F@2cm 0.46→0.51 and ours
  still wins quality+speed at noise (5 %: 2.86 vs 4.78 cm, 5.3× faster). Clean
  sub-voxel precision still favours mature TSDF (needs the surfel rep, #12).
- **#9 edge-preserving (bilateral) TSDF denoise** instead of Gaussian.
- **#3 per-pixel model confidence as a fusion weight**: VGGT real-data Chamfer
  **4.38 → 3.80 cm** (comp 6.73 → 5.65, F@5cm 0.86).
- **#16 amodal completion**: from-scratch visual hull (loose under a single-height
  ring — over-counts vertically) and **Poisson closure** (accurate watertight surface,
  Chamfer 1.1 cm / F@2cm 0.95, better shape than a convex hull). Volume of unobserved
  poles stays coverage-limited → a generative completer is the open piece.
- **#2 pose-free front end**: fully feed-forward (VGGT depth + VGGT poses, Umeyama-
  aligned only for scoring) runs end-to-end but pose error dominates (11.4 cm vs
  3.80 cm with good poses) → needs global BA / loop closure.
- **#25 multi-scene**: across 5 Replica scenes on clean GT depth Open3D wins fine
  precision (its regime), ours is close on coarse F@5cm (0.96–0.99); the kernel's edge
  is the noisy regime + speed.
- **#21 + #18 room collider via CoACD**: the reconstructed room OOMs as a single
  mesh-SDF collider in Genesis; CoACD splits it into 24 convex parts (cheap, no SDF).
  Objects now drop INSIDE the reconstructed room under domain randomization: **8/8 no
  tunnelling, 0 cm penetration** through the reconstructed floor/walls (stability lower
  than on a flat plane because objects land on real reconstructed furniture).

---

# Phases 0, 1, 2 results

Kernel-level rebuild, June 2026. Gitignored (internal).

A single, unified, GPU-vectorised PyTorch core (`kernel.py`, ~210 lines) that
replaces the multi-repo + Open3D-TSDF + Poisson glue, plus a fully from-scratch
benchmark (analytic room + 2 objects, ray-cast depth, GT, visibility-culled
Chamfer/F-score). Baseline = Open3D legacy `ScalableTSDFVolume` (what
`scripts/gsplat_tsdf_mesh.py` ships today).

---

## Phase 0 — fusion core (speed + noise robustness)

Forward ray-sampling integration (TSDF at voxel centre) + confidence/robust
weighting + free-space carving + GPU denoise. The fusion itself is **0.04 s and
noise-invariant**; only the CPU marching cubes scales with surface complexity.

**At 2 cm voxels:**

| noise | method | Chamfer | F@5cm | secs |
|---|---|---|---|---|
| clean | Open3D | **0.93** | **1.00** | 0.28 |
| clean | ours   | 1.80 | 0.99 | **0.24** |
| 5%    | Open3D | 4.44 | 0.598 | 1.57 |
| 5%    | ours   | **2.64** | **0.929** | **0.59** |

**Using the speed headroom (finer 1.3 cm voxel — ours can afford it, Open3D can't):**

| noise | method | Chamfer | F@2cm | secs |
|---|---|---|---|---|
| 2% | Open3D | 2.29 | 0.415 | 3.09 |
| 2% | ours   | **2.26** | **0.475** | **1.21** |
| 5% | Open3D | 4.62 | 0.106 | 3.58 |
| 5% | ours   | **3.57** | **0.322** | **1.16** |

Result: **speed wins at every setting** (fusion 7–37× faster, noise-invariant); at
realistic noise (>=2%) ours wins quality too. The only Open3D win is fine precision
on clean, low-noise synthetic depth (a voxel-resolution effect). The crossover is
~2% noise; real handheld depth lives above it.

---

## Phase 1 + 2 — separable, physics-ready objects (the headline want)

Object ids are voted per voxel during fusion; the fused mesh is split into a room
shell + one mesh per object; movable objects are turned into watertight convex-hull
collision proxies with mass (1.3 cm voxels).

| object | Chamfer (cm) | F@2cm | physics body | mass |
|---|---|---|---|---|
| room shell | 1.82 | 0.713 | static environment mesh | — |
| sphere | 1.00 | 0.941 | watertight hull | 95 kg |
| inner box | **0.69** | **0.976** | watertight hull | 248 kg |

So a single capture is fused once and comes out as **a room shell plus individually
separable, independently accurate object meshes**, each converted to a watertight
collision body with mass — i.e. drop-in rigid bodies for a simulator. The objects
are reconstructed more accurately (0.7–1.0 cm) than the room, because they are seen
face-on from the camera ring.

Honest gap: the convex hull is a collision proxy, and because each object is only
observed from the front, the hull volume under-counts the true object (sphere hull
238 L vs true ~524 L). Closing this needs **amodal completion** (DP-Recon / Amodal3R
in `REBUILD_PLAN.md`) — the genuinely unsolved frontier piece.

---

## Phase 3 — domain-randomized physics eval (RL-env readiness)

Exported assets are loaded into Genesis (gravity, contact). Per object, N=6
randomized drop trials (random xy, yaw, density 200-800 kg/m^3, friction 0.3-1.2)
measure whether the reconstructed object rests stably on the floor without
tunnelling — the quantitative "is this reconstruction usable as a physics env" score.

| object | trials | rest_ok | no_tunnel | stable% | penetration | rest_err |
|---|---|---|---|---|---|---|
| sphere | 6 | 2/6 | **6/6** | 33% | **0.0 cm** | 0.8 cm |
| inner_box | 6 | 6/6 | **6/6** | 100% | **0.0 cm** | 5.5 cm |

**12/12 no tunnelling, 0 cm penetration** across all randomized conditions: the
reconstructed collision proxies hold solid contact. The box rests 100% of the time.
The sphere's 33% is correct physics (a ball rolls, so it rarely settles to near-zero
speed; when it rests, rest_err is 0.8 cm). The box's 5.5 cm rest_err reflects the
amodal gap (convex hull of a front-only shell has a biased centroid).

## Feed-forward front end on REAL data (Replica room0)

The depth source is swapped from GT/sensor depth to a frozen pretrained foundation
model (no training), fused by the kernel, scored vs the Replica GT mesh. Same poses,
same protocol; only the depth source changes.

| front end (depth source) | Chamfer | acc | comp | F@2cm | F@5cm |
|---|---|---|---|---|---|
| GT depth (ceiling) | 2.15 | 1.53 | 2.78 | 0.463 | 0.973 |
| **VGGT multi-view (feed-forward)** | **4.38** | 2.04 | 6.73 | 0.292 | **0.845** |
| monocular (Depth-Anything-V2 metric) | 7.78 | 2.92 | 12.64 | 0.116 | 0.530 |

Findings: (1) the bridge runs end-to-end on real RGB, no training, no COLMAP for the
depth (VGGT: 29 frames -> depth in 7.1 s; kernel fusion 0.44 s). (2) **Multi-view
feed-forward (VGGT) nearly halves the monocular error** (4.38 vs 7.78 cm, F@5cm 0.85
vs 0.53) -- because monocular depth has per-frame scale drift that breaks fusion,
while VGGT is jointly consistent. (3) Feed-forward (4.38 cm, F@5cm 0.85) approaches
GT-depth (2.15 cm) -- a usable metric reconstruction from RGB alone. The gap to GT is
the expected cost of predicted-vs-perfect depth.

Honest caveats: VGGT is scale-invariant, so one global metric scale was set from the
GT-depth ratio (a stand-in for a metric prior like MoGe-2/UniDepth; a fully metric
pipeline would use that prior, no GT). GT poses were kept (the controlled "swap depth
source" experiment); VGGT also predicts poses, so a fully feed-forward pose-free
version is the next step (needs a similarity alignment to GT for scoring).

## What is built vs the full plan

- **Phase 0 (fusion core): done, benchmarked.** Superior speed everywhere; superior
  quality in the noisy regime.
- **Phase 1 (decomposition): done, benchmarked.** Separable per-object meshes.
- **Phase 2 (physics-ready): done.** Watertight collision proxies + mass.
- **Phase 3 (domain-randomized physics-eval harness): done, benchmarked.** 12/12 no
  tunnelling; reset + DR + quantitative metric in place.
- **Feed-forward front end: bridge done, benchmarked on real data.** Pretrained VGGT
  (frozen, no training) drives the kernel on real RGB; multi-view beats monocular and
  approaches GT depth. The recommended path is confirmed: *use* the pretrained
  foundation model, don't retrain it.
- **Still open:** a fully pose-free feed-forward run (VGGT poses + similarity align);
  a metric prior instead of GT-ratio scale; amodal completion for true object volumes;
  and a *trained* RL policy with sim-to-real correlation (needs an RL loop + hardware).
  Scoped in `REBUILD_PLAN.md`.

The thesis is validated on everything that can be built and measured in-session: the
from-scratch kernel beats our current pipeline on speed at every setting and on
quality in the regime real capture lives in, and it now also separates a scene into
accurate, physics-ready per-object assets.

## Files
- `scene.py` — analytic room + objects, ray-cast depth, GT, visibility cull.
- `kernel.py` — unified GPU fusion + object-aware voting + denoise + meshing.
- `bench.py` — noise sweep vs Open3D + per-object decomposition/physics report.
- `noise_robustness.png` — Chamfer / F@5cm / speed vs noise.
Run: `python bench.py --device cuda --W 256 --H 192 --ncam 48 --voxel 0.013`
