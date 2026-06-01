# vid2scene

From a phone video of a room to a metric 3D reconstruction, evaluated against
ground truth and usable as an embodied environment.

A short handheld video of a small indoor room is reconstructed into a metric,
geometrically coherent 3D scene. The project runs three independent
surface-reconstruction methods on the same capture, fuses them with a gated
consensus rule, and evaluates every result against ground-truth meshes. The
resulting metric mesh is then loaded as a navigable environment in which an
agent's first-person view is rendered from the scene.

Built for Humanoid's *From Video to 3D Reconstruction* challenge.

<p align="center">
  <img src="docs/figures/teaser.gif" width="85%" alt="phone capture to metric 3D reconstruction to a robot standing in the room under physics"/><br/>
  <em>The full pipeline in one shot: phone capture &rarr; metric 3D reconstruction &rarr; a robot standing in the reconstructed room under physics.</em>
</p>

<p align="center">
  <img src="docs/figures/replica_compare.gif" width="80%" alt="original frames on the left, reconstruction on the right"/><br/>
  <em>Fidelity check on a benchmarked scene (Replica room0): the original video (left) and our reconstruction (right), rendered along the same camera path. Accuracy against the ground-truth mesh is sub-centimetre.</em>
</p>

## Contributions

The reconstruction backbones are existing state-of-the-art methods. The work
here is the methodology around them.

**1. Multi-method comparison.** Most video-to-3D systems use a single
reconstructor. We run three methodologically distinct methods on the same
capture: [PGSR](https://arxiv.org/abs/2406.06521) (planar Gaussian splatting to a
TSDF mesh), [DN-Splatter](https://arxiv.org/abs/2403.17822) (Gaussian splatting
with monocular depth and normal priors to a TSDF mesh), and
[MonoSDF](https://arxiv.org/abs/2206.00665) (a neural signed-distance field to a
marching-cubes mesh). Comparing them shows where each one fails. Planar
regularization smooths over clutter, the SDF over-inflates, and the
splat-derived TSDF leaves holes.

**2. Gated consensus fusion.** We fuse the meshes by taking one as a backbone and
adding geometry from another only where the backbone has holes, then fitting a
single [screened-Poisson](https://www.cs.jhu.edu/~misha/MyPapers/ToG13.pdf)
surface to the combined points. We avoid volumetric averaging on purpose.
Averaging surfaces with different systematic biases superimposes their errors and
produces doubled or thickened walls. We do not claim a new fusion algorithm. We
claim a simple gating rule and a measured account of when it helps.

**3. Evaluation against ground truth.** We score every mesh and the fusion against
Replica ground-truth meshes using the standard surface-reconstruction protocol
([GO-Surf](https://arxiv.org/abs/2206.14735),
[MonoSDF](https://arxiv.org/abs/2206.00665),
[Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/)):
visibility-culled Accuracy, Completion, Chamfer-L1, F-score at 5 cm, and
Normal-Consistency. The protocol is standard in the reconstruction literature.
Applying it to a casual phone-capture pipeline, where rendering quality is
usually the only number reported, is uncommon.

**4. Metric scale.** Monocular reconstruction recovers shape only up to an unknown
global scale factor. We keep the reconstruction in real-world units, because the
questions that matter for a robot, such as clearance, collision margins, and
reachability, are only well-posed at true scale. A scale-free mesh can render
correctly and still give wrong answers to all of them.

**5. Embodied use.** The metric mesh is used as a robot environment two ways. In
Habitat it becomes a navmesh that an agent walks. In Genesis it becomes a rigid-
body physics collider: a dropped sphere falls under gravity and comes to rest on
the reconstructed floor at metric scale (within 2 cm of floor + radius), and a
Go2 quadruped spawns and holds a stable stance on the same floor under PD
control. This shows the reconstruction supports real physics at true scale, not
only rendering. It builds on the real-to-sim-via-splatting line of work
([EmbodiedSplat](https://arxiv.org/abs/2509.17430),
[VR-Robo](https://arxiv.org/abs/2502.01536),
[GaussGym](https://arxiv.org/abs/2510.15352)) as a systems integration, not a new
simulator. What remains is a trained locomotion policy for forward walking and
the photoreal Genesis camera; see [docs/PHASE2_GENESIS.md](docs/PHASE2_GENESIS.md)
and [scripts/genesis/](scripts/genesis).

### The fusion, precisely

Given a trusted backbone mesh `B`, a donor mesh `D`, and a distance threshold `tau`:

```
keep = { d in vertices(D) : dist(d, nearest vertex of B) > tau }   # donor fills only B's holes
S    = oriented, coloured points of B  ∪  keep
M    = ScreenedPoisson(S), then trim the lowest-density vertices and keep the
       largest connected component
```

`tau` controls how aggressively the donor fills (smaller `tau` borrows more).
The gate is what prevents the doubled-surface artifact: where `B` already has a
surface, the donor is ignored, so two biased surfaces are never averaged.
Implementation: [`src/vid2scene/fuse/consensus.py`](src/vid2scene/fuse/consensus.py).

## Results

Table 1 reports reconstruction accuracy against Replica ground-truth meshes,
averaged over five scenes (room0-2, office0-1). All meshes are visibility-culled
before scoring. Distances are in centimetres; F-score uses a 5 cm threshold.
Reproduce with `make benchmark`, which reads the bundled metrics in
`runs/replica_eval/`.

**Table 1. Five-scene average.**

| Method | Accuracy ↓ | Completion ↓ | Chamfer-L1 ↓ | Normal-C ↑ | F-score ↑ |
|---|---|---|---|---|---|
| PGSR | 1.13 | 7.60 | 4.37 | 0.938 | 0.898 |
| DN-Splatter | 0.57 | 6.14 | 3.36 | 0.965 | 0.936 |
| Consensus (fusion) | 1.07 | 6.47 | 3.77 | 0.944 | 0.913 |

DN-Splatter is the most accurate single method, with sub-centimetre accuracy on
every scene. The consensus fusion does not beat the best single method on these
scenes, most of which have near-complete camera coverage. What the fusion does is
improve the mesh it is built on, and the improvement grows with scene difficulty.
Table 2 gives the Chamfer-L1 per scene.

**Table 2. Chamfer-L1 per scene (cm).**

| Scene | PGSR | DN-Splatter | Consensus |
|---|---|---|---|
| room0 | 1.50 | 0.62 | 1.48 |
| room1 | 1.21 | 0.88 | 1.21 |
| room2 | 4.29 | 1.97 | 2.34 |
| office0 | 7.72 | 6.50 | 6.85 |
| office1 | 7.11 | 6.81 | 6.97 |
| average | 4.37 | 3.36 | 3.77 |

On the cluttered `room2` the fusion lowers the PGSR backbone's Chamfer-L1 from
4.29 to 2.34 cm. On `office1`, where camera coverage is poorest, a variant that
uses DN-Splatter as the backbone reaches 6.48 cm against DN-Splatter's own
6.81 cm. The fusion trades a small amount of accuracy for better completion,
which helps when coverage is incomplete and is roughly neutral when coverage is
already good. The full per-scene metrics and the backbone ablation are in
[docs/BENCHMARK.md](docs/BENCHMARK.md).

<p align="center">
  <img src="docs/figures/benchmark_bars.png" width="88%" alt="grouped bar chart of Chamfer-L1 and F-score for PGSR, DN-Splatter and the consensus fusion on Replica and on a real iPhone capture"/><br/>
  <em>Tables 1 and 3 at a glance: Chamfer-L1 (lower is better) and F-score@5cm (higher is better) for the three methods, on the Replica average and on the real iPhone capture.</em>
</p>

The standard surface-reconstruction error visualization makes the per-method
differences legible. Each mesh vertex is coloured by its distance to the
ground-truth mesh on the cluttered `room2`, clamped at 5 cm:

<p align="center">
  <img src="docs/figures/error_heatmap.png" width="100%" alt="per-vertex distance-to-ground-truth error, colormapped, for PGSR, DN-Splatter and the consensus fusion on room2"/><br/>
  <em>Per-vertex error against the GT mesh (turbo colormap, 0-5 cm). Blue is accurate; warmer is further from ground truth. DN-Splatter is cleanest overall; the fusion concentrates its remaining error in the cluttered, low-coverage regions.</em>
</p>

All five benchmark scenes, input video against reconstruction, played along the
same camera path. Each pair is the original capture (left) and our reconstruction
(right). Top row: `room0`, `room1`, `room2`; bottom row: `office0`, `office1`.

<p align="center">
  <img src="docs/figures/benchmark_grid.gif" width="100%" alt="grid of all five Replica benchmark scenes, each showing the input video next to the reconstruction along the same trajectory"/>
</p>

The reconstructions track the input closely on the rooms; the offices are dimmer
scenes with poorer camera coverage, which is where the larger Chamfer-L1 in
Table 2 comes from.

For reference, the Gaussian splat used as the appearance model reaches 32.3 PSNR
on the Mip-NeRF 360 `room` scene, within 0.7 dB of the best published result on
that scene. This is a check that the front end is competitive, not a headline
result.

### On a real iPhone capture

The scenes above are rendered. We also ran the full pipeline on a genuine
handheld iPhone capture with a Faro laser-scan ground-truth mesh (MuSHRoom
`coffee_room`), scored with the same protocol.

**Table 3. Real iPhone capture (MuSHRoom `coffee_room`).**

| Method | Chamfer-L1 ↓ (cm) | F-score ↑ |
|---|---|---|
| PGSR | 4.98 | 0.785 |
| **DN-Splatter** | **1.91** | **0.946** |
| Consensus | 4.69 | 0.796 |

The findings transfer to real data: DN-Splatter reconstructs the room to about
2 cm against the laser ground truth, and the fusion again improves its PGSR
backbone (4.98 → 4.69 cm) without beating the best single method. Errors are
higher than on the rendered scenes, as expected for real capture. Full numbers in
[docs/BENCHMARK.md](docs/BENCHMARK.md).

<p align="center">
  <img src="docs/figures/mushroom_fly.gif" width="42%" alt="reconstruction of a real iPhone capture (MuSHRoom coffee_room)"/><br/>
  <em>Reconstruction of the real iPhone <code>coffee_room</code> capture. Rougher than the rendered scenes, and validated at roughly 2 cm against the Faro ground-truth mesh.</em>
</p>

## How it works

<p align="center">
  <img src="docs/figures/pipeline.png" width="92%" alt="system pipeline: video to ingest to three reconstructors to consensus fusion to benchmark and embodiment"/>
</p>

```
phone video
  -> ingest        keyframe selection (blur gate + parallax spacing)      [CPU]
  -> reconstruct   PGSR, DN-Splatter, MonoSDF                             [GPU]
  -> fuse          gated consensus (backbone + donor-in-holes)            [CPU]
  -> benchmark     visibility-culled Chamfer / F-score vs GT mesh         [GPU+CPU]
  -> embodied      metric mesh -> Habitat navmesh -> splat-rendered agent [CPU + sim]
```

The CPU stages install with `pip install -e .` and run on a laptop. The three
reconstructors run on a CUDA host, driven by [`scripts/remote/`](scripts/remote).
The full repo map and per-stage code locations are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```bash
make install            # CPU stages
make benchmark          # reproduce the ground-truth table from bundled metrics
make viewer             # serve the interactive viewers at http://localhost:8765

# fuse any two coloured meshes (CPU)
vid2scene fuse --backbone pgsr.ply --donor dn.ply --out consensus.ply
# export the metric mesh as a sim-ready collider
vid2scene embodied --mesh consensus.ply --out room_sim.glb --scene-json scene.json
```

### Interactive viewers

- `viewer/replica_room0.html` toggles the reconstruction against the Replica ground-truth mesh.
- `viewer/mesh_compare.html` shows the three methods and the fusion aligned in one frame.
- `viewer/splat_ref.html` shows the reference Gaussian splat.

<p align="center">
  <img src="docs/figures/go2_walk.gif" width="62%" alt="Go2 quadruped traversing the reconstructed room, fixed camera"/><br/>
  <em>Genesis backend: a Go2 quadruped traverses the phone-scanned room at metric scale (fixed camera). The base follows a planned path over the open floor with a trot gait; this is a kinematic traversal, as physics-driven locomotion needs a trained policy (see Limitations). Rigid-body physics is validated separately below.</em>
</p>

<p align="center">
  <img src="docs/figures/genesis_physics.png" width="92%" alt="reconstructed room in Genesis with a sphere settling on the floor"/><br/>
  <em>The reconstructed room as a rigid-body collider. A dropped sphere falls under gravity and rests on the floor at metric scale (right), validating import, scale, collision, and gravity.</em>
</p>

<p align="center">
  <img src="docs/figures/topdown_path.png" width="48%" alt="agent navigation path over the reconstructed room, top-down"/><br/>
  <em>Habitat backend: the metric mesh as a navmesh, with a recorded agent trajectory (top-down).</em>
</p>

## Design choices

- **Three methods, because one cannot self-validate.** Comparing distinct
  reconstruction families against ground truth turns a visual impression into a
  measurement and shows where each one breaks.
- **Gating rather than averaging.** The fusion borrows geometry only in backbone
  holes, which avoids the doubled-surface artifact that averaging biased meshes
  produces. The benefit is conditional, and the condition is reported.
- **Metric units from the start,** because the embodied stage depends on them.
- **Explicit about coverage.** Surfaces the camera never observed cannot be
  recovered faithfully. We either fill them and say so or leave them, and the
  capture-coverage diagnosis is shown in [docs/figures](docs/figures).
- **Two environments by design.** A single `pip install` does not reproduce hours
  of GPU training, so the training drivers live in
  [`scripts/remote/`](scripts/remote) and are documented rather than hidden.

## Limitations

- **Coverage.** Surfaces the camera never observed cannot be recovered. The
  pipeline either fills them by interpolation (screened Poisson) or leaves them
  open, and the choice is stated per result. This is a property of single-pass
  capture, not of any one method. The per-vertex error map above shows where this
  bites: residual error concentrates in the regions the camera saw least.
- **The fusion is conditional.** The gated consensus improves its backbone and
  helps most on cluttered or low-coverage scenes, but it does not beat the
  strongest single method on clean, fully-observed scenes. It trades a little
  accuracy for completeness rather than improving both.
- **Metric scale depends on the input.** Real-world units come from the
  capture's poses or depth priors (ARKit or sensor depth). A purely monocular
  capture with no scale cue is recovered only up to a global scale factor.
- **Benchmark spans synthetic and real, but is small.** Ground-truth numbers are
  on five rendered Replica scenes and one real iPhone capture (MuSHRoom, Faro-laser
  ground truth). Real-capture errors are higher, as expected. A larger real-data
  sweep would strengthen the claims further.
- **Geometry only.** Semantic labelling is out of scope in this version.
- **Embodiment.** Habitat navigation and Genesis rigid-body physics (object drop,
  Go2 stand) are implemented. Forward locomotion needs a trained Go2 policy, and
  the photoreal Genesis camera needs CUDA 12.9; both are open
  (`docs/PHASE2_GENESIS.md`).

## Repository layout

```
src/vid2scene/   ingest, fuse, benchmark, viz, embodied      (pip-installable CPU stages + CLI)
scripts/remote/  GPU reconstruction and benchmark backend    (PGSR / DN-Splatter / MonoSDF / fusion / eval)
scripts/         TSDF meshing, Habitat navmesh + path, photoreal rendering
viewer/          three.js and Gaussian-splat web viewers
docs/            ARCHITECTURE, BENCHMARK, PHASE2_GENESIS, FINDINGS
runs/            outputs and the bundled ground-truth metrics (replica_eval/)
```

Start at [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## References

Reconstruction and surface methods:
- Chen et al. *PGSR: Planar-based Gaussian Splatting Reconstruction.* TVCG 2024. [arXiv:2406.06521](https://arxiv.org/abs/2406.06521)
- Turkulainen et al. *DN-Splatter: Depth and Normal Priors for Gaussian Splatting.* WACV 2025. [arXiv:2403.17822](https://arxiv.org/abs/2403.17822)
- Yu et al. *MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction.* NeurIPS 2022. [arXiv:2206.00665](https://arxiv.org/abs/2206.00665)
- Kazhdan and Hoppe. *Screened Poisson Surface Reconstruction.* ACM ToG 2013.
- Wang et al. *GO-Surf.* 3DV 2022. [arXiv:2206.14735](https://arxiv.org/abs/2206.14735)
- Li et al. *Neuralangelo.* CVPR 2023.

Embodied / real-to-sim:
- Khanna et al. *EmbodiedSplat.* ICCV 2025. [arXiv:2509.17430](https://arxiv.org/abs/2509.17430)
- *VR-Robo.* RA-L 2025. [arXiv:2502.01536](https://arxiv.org/abs/2502.01536)
- *GaussGym.* 2025. [arXiv:2510.15352](https://arxiv.org/abs/2510.15352)
- Genesis-Embodied-AI. *Genesis.* https://github.com/Genesis-Embodied-AI/genesis-world

Datasets:
- Straub et al. *The Replica Dataset.* 2019.
- Ren et al. *MuSHRoom: Multi-Sensor Hybrid Room Dataset.* WACV 2024.
- Barron et al. *Mip-NeRF 360.* CVPR 2022.

