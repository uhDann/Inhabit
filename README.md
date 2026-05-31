# vid2scene

**From a phone video of a room to a metric 3D reconstruction that is quantitatively evaluated against ground truth and usable as an embodied environment.**

A short handheld video of a small indoor room is reconstructed into a metric,
geometrically coherent 3D scene. Rather than stopping at a single reconstruction
and a viewer, the project does three things that are uncommon for casual,
phone-capture pipelines: it runs and **compares three independent
surface-reconstruction methods** on the same capture, **fuses** them with a
gated consensus rule, and **evaluates every result against ground-truth meshes**
under the standard visibility-culled protocol. The resulting metric mesh is then
carried into an **embodied stage** — a navigable environment in which an agent's
first-person view is rendered from the scene.

Built for Humanoid's *From Video to 3D Reconstruction* challenge.

<p align="center">
  <img src="docs/figures/compare.gif" width="80%" alt="left: original video; right: reconstruction rendered along the same path"/><br/>
  <em>Left: original phone video. Right: the reconstruction, rendered along the same camera path.</em>
</p>

---

## What the project does, and what is actually new

The reconstruction backbones are off-the-shelf and state of the art; they are
not the contribution. The contribution is the methodology wrapped around them.

**1. A systematic multi-method comparison, not a single pipeline.**
Most video-to-3D systems commit to one reconstructor. We run three
methodologically distinct surface-reconstruction families on the same capture —
[PGSR](https://arxiv.org/abs/2406.06521) (planar Gaussian splatting → TSDF mesh),
[DN-Splatter](https://arxiv.org/abs/2403.17822) (Gaussian splatting with
monocular depth/normal priors → TSDF mesh), and
[MonoSDF](https://arxiv.org/abs/2206.00665) (neural signed-distance field →
marching cubes) — which exposes exactly where each fails: planar regularization
smooths clutter, the SDF over-inflates, the splat-derived TSDF leaves holes.

**2. A gated consensus fusion that avoids a known failure mode.**
We fuse the meshes with a trusted backbone plus donor geometry **only where the
backbone has holes** ([screened Poisson](https://www.cs.jhu.edu/~misha/MyPapers/ToG13.pdf)
over the gated, oriented point union). This is deliberately *not* volumetric
averaging: averaging independently-biased surfaces is known to superimpose their
errors and produce doubled or thickened walls. We do not claim a new fusion
algorithm; we claim a simple, principled gating and a rigorous account of *when*
it helps.

**3. Quantitative evaluation against ground truth — rare in this setting.**
We score every individual mesh and the fusion against Replica ground-truth
meshes using the standard surface-reconstruction protocol
([GO-Surf](https://arxiv.org/abs/2206.14735),
[MonoSDF](https://arxiv.org/abs/2206.00665),
[Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/)):
visibility-culled Accuracy, Completion, Chamfer-L1, F-score@5 cm, and
Normal-Consistency. The protocol is the academic standard; applying it to a
casual phone-capture pipeline — where rendering quality is usually the only
reported number — is what is uncommon here.

**4. Metric scale carried through to embodiment.**
Monocular reconstruction is scale-ambiguous by construction: correctly shaped
but not in real units. We keep the reconstruction *metric*, because the embodied
questions that matter for a robot — clearance, collision margin, reachability,
"can a 1.6 m agent pass under this table" — are only well-posed at true scale. A
scale-free mesh can render perfectly and still answer all of them wrong.

**5. An embodied demonstration, not just a mesh.**
The metric mesh becomes a navigable environment (Habitat navmesh) in which an
agent walks the room and its first-person view is re-rendered from the Gaussian
splat. This is a systems-integration demonstration built on the real-to-sim-via-
splatting line of work ([EmbodiedSplat](https://arxiv.org/abs/2509.17430),
[VR-Robo](https://arxiv.org/abs/2502.01536),
[GaussGym](https://arxiv.org/abs/2510.15352)) — not a new simulator. A
Genesis-based physics-and-RL upgrade is **designed but not yet implemented**
([docs/PHASE2_GENESIS.md](docs/PHASE2_GENESIS.md)).

---

## Results

### Reconstruction quality against ground truth

Replica, five scenes (room0–2, office0–1), visibility-culled, distances in cm,
F-score at 5 cm. Reproduce locally with `make benchmark` (reads the bundled
metrics in `runs/replica_eval/`).

| Method | Accuracy ↓ | Completion ↓ | Chamfer-L1 ↓ | F-score ↑ |
|---|---|---|---|---|
| PGSR | 1.13 | 7.60 | 4.37 | 0.898 |
| DN-Splatter | 0.57 | 6.14 | 3.36 | 0.936 |
| Consensus (fusion) | 1.07 | 6.47 | 3.77 | 0.913 |

**Reading the table honestly.** DN-Splatter is the strongest single method on
every scene; the reconstruction is sub-centimetre in accuracy. The consensus
fusion **does not beat the best single method on these mostly clean scenes** —
and we report that as a finding rather than hide it. What the fusion does
reliably is improve *its own backbone*, by a margin that grows with scene
difficulty: on the cluttered `room2` it cuts the PGSR backbone's Chamfer-L1 from
4.29 to 2.34 cm (−45%), and on `office1` a DN-backbone variant edges past
DN-Splatter alone. In other words, the fusion trades a little accuracy for more
completeness, which is a net gain precisely where coverage is poor. Full
per-scene tables, the protocol, and the ablation are in
[docs/BENCHMARK.md](docs/BENCHMARK.md).

<p align="center">
  <img src="docs/figures/replica_compare.gif" width="78%" alt="Replica room0: original render (left) vs our reconstruction (right) along the capture path"/><br/>
  <em>A benchmarked scene (Replica room0): original (left) vs our reconstruction (right), along the capture path. The reconstruction is sub-centimetre in accuracy against the ground-truth mesh.</em>
</p>

The reference Gaussian splat reaches PSNR 32.3 on the Mip-NeRF 360 `room` scene,
within 0.7 dB of the best published result on that scene — a sanity check that
the front-end is competitive, not a headline number.

---

## How it works

```
phone video
  → ingest        keyframe selection (blur gate + parallax spacing)      [CPU]
  → reconstruct   PGSR · DN-Splatter · MonoSDF                            [GPU]
  → fuse          gated consensus (backbone + donor-in-holes)            [CPU]
  → benchmark     visibility-culled Chamfer / F-score vs GT mesh         [GPU+CPU]
  → embodied      metric mesh → Habitat navmesh → splat-rendered agent   [CPU export + sim]
```

The CPU stages install with `pip install -e .` and run on a laptop; the three
reconstructors run on a CUDA host, driven by [`scripts/remote/`](scripts/remote).
Full repo map and per-stage code locations: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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

- `viewer/replica_room0.html` — reconstruction toggled against the Replica ground-truth mesh.
- `viewer/mesh_compare.html` — the three methods and the fusion, aligned in one frame.
- `viewer/splat_ref.html` — the reference Gaussian splat.

<p align="center">
  <img src="docs/figures/explore_agent.gif" width="62%" alt="agent navigating the reconstructed room"/><br/>
  <em>An agent navigates the metric reconstruction; its view is rendered from the Gaussian splat.</em>
</p>

---

## Design choices and tradeoffs

- **Three methods, because one cannot self-validate.** Comparing distinct
  reconstruction families against ground truth turns "looks correct" into a
  measurement and shows where each one breaks.
- **Gating, not averaging.** The fusion borrows donor geometry only in backbone
  holes, to avoid the doubled-surface artifact that averaging biased meshes
  produces. Its benefit is conditional, and we characterize the condition.
- **Metric from the start.** Real units are a prerequisite for the embodied
  stage, not a finishing touch.
- **Honest about coverage.** Surfaces the camera never observed cannot be
  recovered faithfully; we either fill them and say so, or leave them, and show
  the capture-coverage diagnosis ([docs/figures](docs/figures)).
- **Two environments on purpose.** A single `pip install` does not reproduce
  hours of GPU training; the exact training drivers live in
  [`scripts/remote/`](scripts/remote) and are documented rather than hidden.

## Repository layout

```
src/vid2scene/   ingest · fuse · benchmark · viz · embodied   (pip-installable CPU stages + CLI)
scripts/remote/  GPU reconstruction and benchmark backend     (PGSR / DN-Splatter / MonoSDF / fusion / eval)
scripts/         TSDF meshing · Habitat navmesh + path · photoreal rendering
viewer/          three.js and Gaussian-splat web viewers
docs/            ARCHITECTURE · BENCHMARK · PHASE2_GENESIS · FINDINGS
runs/            outputs and the bundled ground-truth metrics (replica_eval/)
```

Start at [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
