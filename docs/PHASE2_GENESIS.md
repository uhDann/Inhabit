# Phase 2 — Genesis: turning the room scan into a real robot RL environment

**Status:** Steps 1 and 2 (physics) are now IMPLEMENTED; the rest remains proposed. This document scoped the upgrade and is kept for the full plan; the implemented part is summarised here.

> **Implemented (`scripts/genesis/`, evidence in `scripts/genesis/outputs/`):**
> - genesis-world 1.0.0 initialises on the RTX 4070 Ti SUPER (`gs.cuda`) and CPU. No nvcc / source build needed (bundled `quadrants` JIT). Install gotcha: pin `numpy<2` (genesis pulls numpy 2.x, which breaks the torch 2.2 ABI).
> - **Step 1 — object drop:** the reconstructed Replica room0 mesh, gravity-aligned (RANSAC floor → floor at z=0), is loaded as a static collider. A rigid sphere (r=0.12 m) dropped from 0.6 m falls under gravity and rests at base z ≈ 0.098 m (floor + radius, within 2 cm) — no tunnelling. Validates mesh import + metric scale + collision + gravity.
> - **Step 2 — Go2 quadruped:** the Genesis-shipped Go2 URDF (18 DOF) spawns at z=0.42 m and settles to a stable standing base height z ≈ 0.29 m, held for 4 s under PD control. Forward walking produces no net travel without a trained RL policy.
>
> **Still open:** a trained Go2 locomotion policy for walking (the policy is in Genesis's GitHub repo, not the pip wheel); the photoreal `genesis-nyx` 3DGS camera (needs CUDA 12.9; our box is 12.1/12.6).

---

The full migration plan below is retained as the roadmap.

## 1. Motivation

Phase 1 nailed the geometry side: a phone-style capture becomes a textured Gaussian splat, a watertight-ish mesh, a Habitat navmesh, and a photoreal first-person video of an agent autonomously walking the room. But the embodied side is intentionally minimal — Habitat-Sim gives us navmesh-based locomotion for a point/cylinder agent, no physics, no actual robot body, and our "photoreal" video is a hack: we record the agent's poses, then re-render each frame separately through the gsplat trainer's rasterizer.

The natural next step is: **turn every room scan into a real-physics RL environment usable for robot training**, with a real robot body (e.g. a Go2 quadruped or a Franka arm), real contact physics, and a photoreal first-person camera *baked into the simulator* rather than rendered post-hoc.

The Genesis stack, released by Genesis-Embodied-AI, is the cleanest path. It is essentially the unified, more capable replacement for the Habitat + gsplat-render-after split we have now.

## 2. What Genesis is (three pieces, all Apache-2.0)

| Repo | What it is | Role for us |
|---|---|---|
| [`genesis-world`](https://github.com/Genesis-Embodied-AI/genesis-world) | Multi-physics simulator with a unified Python API. Ingests **URDF, MJCF, OBJ, GLB, USD**; rigid + FEM + MPM + PBD/SPH + IPC solvers; runtime renderer supports **3D Gaussian splats**. Ships drivable robots (Franka, Go2, drones). | Hosts the room (our GLB/mesh) and the robot body; runs the physics step and the RL env. |
| [`genesis-nyx`](https://github.com/Genesis-Embodied-AI/genesis-nyx) | GPU path-traced renderer that plugs into `genesis-world` as a **camera sensor**. PBR materials, HDRI lighting, **native 3DGS rendering**, per-pixel object IDs. CUDA 12.9+, prebuilt wheels for Linux + Windows / Python 3.10–3.13. | The robot's onboard camera renders **directly from our trained Gaussian splat** — no record-then-replay hack. |
| [`quadrants`](https://github.com/Genesis-Embodied-AI/quadrants) | Python → parallel-kernel compiler forked from Taichi (June 2025). Targets CUDA / Vulkan / Metal / ROCm / CPU. Underpins the speed of `genesis-world`. | Infra. We don't write `@qd.kernel` code ourselves; this just makes the sim fast. |

## 3. Why this is a strict upgrade over Habitat + gsplat-render

| Capability | Phase 1 (Habitat + gsplat-render) | Phase 2 (Genesis + Nyx) |
|---|---|---|
| Agent body | Point/cylinder, navmesh-locked | Real robot (Go2 quadruped, Franka, etc.) with URDF + actuators |
| Contact physics | None (kinematic on navmesh) | Real rigid contacts; FEM/MPM if we ever care |
| Camera observation | Record path → re-render frames offline through gsplat trainer | Camera sensor in the sim, rendered **live from our 3DGS** by Nyx |
| Tasks | Point-to-point navigation | Navigation, exploration, manipulation, locomotion-on-uneven-floor |
| Reset / RL loop | Hand-rolled around `sim.step` | First-class `env.step` / `env.reset` semantics |
| Sim2Real story | "We can render the agent's view" | "We trained a Go2 policy in a photoreal scan of *this specific* room" |

## 4. Concrete migration plan

Phases below assume Phase 1's artefacts as input (everything already exists under `runs/mip360_room/`):

- `ckpt_clean.pt`, `room_clean2.ply` — the cleaned reference splat (PSNR 32.3 hero)
- `runs/dnsplat_room/mesh/room_final.ply` — the dense textured DN-Splatter mesh (5.7 M verts, ~the actual room)
- `runs/monosdf_room/monosdf_mesh_watertight.ply` — the watertight SDF mesh (post tight-bound retrain: clean watertight shell)
- `scene.json` — the source→Habitat alignment transform `M_s2h`

### Step 1 — Get `genesis-world` running with our room as the static GLB

1. New conda env `genesis` (Python 3.10, torch matching the Genesis wheel).
2. `pip install genesis-world` (Apache-2.0).
3. Convert the DN-Splatter mesh to GLB with the Phase-1 transform already applied (we already do this for Habitat — see `scripts/export_habitat_glb.py`; gravity-aligned, Y-up).
4. Smallest possible scene: load the GLB as a static collider, drop a sphere agent, step physics, render a debug image. This validates the geometry import end-to-end.

**Success criterion:** sphere agent sits on the floor under gravity instead of falling through. (Catches GLB scale / orientation bugs upfront.)

### Step 2 — Swap the agent for a Go2 quadruped on our floor

1. Use Genesis's bundled Go2 URDF.
2. Spawn at the COLMAP camera centroid (where Phase 1 records the agent's starting pose).
3. Run a simple stand-still + walk-forward policy from Genesis examples; verify navmesh-free locomotion across the textured floor.
4. Compare to Phase 1's Habitat greedy follower video; document differences.

**Success criterion:** Go2 walks across the room without tunneling through the mesh.

### Step 3 — Plug the Gaussian splat into Nyx as the camera sensor

1. Upgrade the env to CUDA 12.9 (Nyx requirement). Cleanest is a fresh `genesis-nyx` env with the CUDA-12.9-targeted wheel; share a conda CUDA toolchain like we did for MonoSDF.
2. Attach a Nyx camera sensor to the Go2's head.
3. Point Nyx at our `room_clean2.ply` so the sensor renders the gaussian splat (not the mesh).
4. Record a walkthrough video — the robot's view through our splat, generated *live* by the sim.

**Success criterion:** the live Nyx render of the gsplat looks indistinguishable from the offline `gsplat_render_path.py` output we produced in Phase 1.

### Step 4 — Wire up a basic navigation RL task

1. Define a Genesis env: observation = Nyx RGB + Go2 joint state; action = Go2 joint targets; reward = distance-to-goal − collision penalty − energy.
2. Train PPO (Genesis ships RL helpers) for ~10 M steps; goals sampled from the central navigable region (same idea as Phase 1's `central_point` sampler).
3. Roll out the trained policy, record a video.

**Success criterion:** a policy that consistently reaches central goals in the reconstructed room, with the recorded video showing the photoreal view (Nyx + our splat) of the trained robot's walk.

### Step 5 — Generalize beyond Mip-NeRF 360 `room`

The whole point: this should work for any scan, not just the dataset room.

1. Take one of the user's own phone captures (PropertyScanAI-era videos under `examples/`).
2. Re-run the Phase 1 pipeline to produce splat + DN-Splatter mesh + tight MonoSDF watertight mesh + scene.json.
3. Drop the artefacts into the Step-1 → Step-4 stack with zero code changes.
4. Train a Go2 policy in *that* room.

This is the "phone scan → explorable RL environment for humanoids" pitch, end-to-end.

## 5. Gotchas to budget for

- **CUDA 12.9 for Nyx.** Our box has system runtime-only CUDA 12.6 (no nvcc) and we installed conda CUDA 12.1 for MonoSDF/tcnn. Nyx wants 12.9. Plan: separate conda env with cuda-toolkit 12.9 from the `nvidia/label/cuda-12.9.0` channel; same minimal-components trick we used for the MonoSDF env to avoid the `cuda-gdb` post-link breakage.
- **Coordinate frames.** Phase 1 already documents the OpenCV / OpenGL / Habitat axis swaps and the `M_s2h` source→Habitat transform in `scene.json`. The same care is needed for Genesis (it appears to follow MuJoCo-style Z-up; needs verification). Most likely an `Rx(+90)` like the Habitat import, but to be confirmed.
- **GLB scale.** Genesis URDFs are metric. Our DN-Splatter mesh is in nerfstudio's normalized frame (apply the inverse `dataparser_transforms.json` rotation, then the COLMAP scale). The MonoSDF mesh after the world-space transform is metric in COLMAP units; the scale factor between COLMAP units and real metres depends on the scan and may need calibration (place a known-size object in the scene, or trust the post-tightening `scale_mat`).
- **Splat alignment vs mesh alignment.** Nyx renders the splat; physics uses the mesh. They must be in the *same world frame* for the rendered image to match the simulated camera pose. We already have `scene.json` doing exactly this between gsplat space and Habitat space; the same transform (or its analogue for Genesis) applies.

## 6. Honest comparison vs. building on what we have

We could keep iterating Phase 1 instead: smoother MonoSDF (tighter scene bound is launching now), Manhattan-SDF or PGSR for cleaner walls, a richer Habitat task, etc. That's lower risk and produces a better Phase-1 demo.

Phase 2 (Genesis) is higher risk (multiple new dependencies, CUDA 12.9 upgrade, new conventions to debug) but is the **only** path to:

- A real robot body that we can later transfer to a real robot.
- Photoreal camera observations rendered **inside** the sim loop (not post-hoc).
- A unified RL env that ships with the project rather than being a hand-rolled stack of Habitat-Sim + ffmpeg + offline gsplat rendering.

Strong recommendation for the writeup narrative: **Phase 1 stands on its own (it's a complete scan-to-explorable-room demo with a working agent video)**, and we cite Phase 2 as the planned extension that this work directly enables. If GPU time permits, we land Step 1–Step 3 of Phase 2 as a stretch demo.

## 7. References

- Genesis-world: https://github.com/Genesis-Embodied-AI/genesis-world
- Genesis-nyx (path-traced renderer + camera sensor + 3DGS): https://github.com/Genesis-Embodied-AI/genesis-nyx
- Quadrants (Taichi-derived compiler): https://github.com/Genesis-Embodied-AI/quadrants
- Phase-1 running notes: [FINDINGS.md](FINDINGS.md)
- Research context (real2sim from room scans, EmbodiedSplat / GaussGym / VR-Robo / Habitat-GS): see `MEMORY.md` → `humanoid_real2sim_research.md`
- Watertight-mesh investigation that produced the Phase-1 meshes: `MEMORY.md` → `watertight_mesh_direction.md`

