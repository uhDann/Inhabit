# vid2scene — Findings & Running Notes

_Project: submission for the Humanoid "Perception & Spatial AI" intern challenge (video → geometrically coherent 3D of a small indoor room)._

> **Companion docs**
> - [BENCHMARK.md](BENCHMARK.md) — quantitative benchmark protocol (Replica GT-mesh; Chamfer/F-score; the consensus-fusion hypothesis) and results.
> - [PHASE2_GENESIS.md](PHASE2_GENESIS.md) — proposal to upgrade the embodied side from Habitat to the Genesis stack (genesis-world + genesis-nyx + quadrants), turning the scan into a real-robot RL environment with photoreal in-sim camera observations.

---

## 1. Input quality diagnosis — is the video bad, or the pipeline?

**Verdict: it's mostly the INPUT. The pipeline works correctly; the capture is the main limiter.** See `docs/figures/input_quality.png`.

Source video (`MapAnything_Test/reconstruction/video.mov`):
- **960×720, 3 fps, 1042 frames, 347 s** — i.e. **low resolution** (downsampled 2× from the phone's native 1920×1440) and a long, slow handheld pan.
- Frame sharpness (variance-of-Laplacian on a 320-wide gray, sampled): video **min 1, median ~338, max 6281** → huge variance, a real motion-blur tail (min 1 = badly blurred frames). Our ingest already drops ~36% of frames for blur.
- Native **photos/ are 1920×1440** (88 of them) — 4× the pixels, visibly more detail than the video frames.

What the frames show (and why it's hard for *any* method):
1. **Low resolution** (960×720) caps achievable detail.
2. **Large textureless white walls** — the dominant failure mode. Blank walls give no features/ambiguous depth, so reconstructions go soft/blobby on flat surfaces. This hurts COLMAP, NeRF, and splatting alike.
3. **Motion blur** on a fraction of frames (handheld).
4. The room is a fairly empty bedroom (plain surfaces, few textured objects).

**Implication:** the single biggest quality lever is a **better capture** (≥1080p, slower/steadier to kill blur, more textured content, and ideally *orbiting* objects rather than only panning walls). With the existing data, the cheapest win is using the **native 1920×1440 photos** instead of the 960×720 video frames. The pipeline itself is producing a coherent metric room — it is not the bottleneck.

Also note the "blobs from outside the room" problem in the splat viewer is **capture coverage**, not a bug: a wall-pan only constrains geometry from along that path, so free-orbit (esp. from outside) shows unobserved angles as blobs. Every method (COLMAP/NeRF/splat) shares this.

---

## 2. What works (pipeline status)

End-to-end, on free UCL GPU compute (trailbreaker):
- **Ingest / frame validation** (`vid2scene ingest`) — blur gate + motion-spaced keyframes + HTML report. CPU. ✓
- **Geometry** — MapAnything (Apache weights, **metric**), pose-free **and** pose-assisted (ARKit poses+intrinsics, ARKit→OpenCV convert). ✓ Produces a coherent metric room; pose-assisted clearly better than pose-free (the A/B).
- **Point cloud + viewer** — colored PLY (confidence-masked, flyer-cropped) + three.js web viewer (`viewer/index.html`). ✓
- **Gaussian splat** — gsplat with `DefaultStrategy` densification (100k→472k), L1+SSIM, exports SuperSplat-compatible `.ply` + a GaussianSplats3D web viewer (`viewer/splat.html`). ✓ Looks good *along the capture path*, blobby off-trajectory (capture limit).
- **Offline renderer** — `scripts/render_pointcloud.py` (numpy+cv2 montage/turntable, no GPU/browser). ✓

Not done yet (the parts that get graded): README, RESULTS/metrics, hosted demo, optional semantics. Proposed but not built: **TSDF mesh** (coherent from all angles), **pose A/B numbers** via `evo`.

---

## 3. Design differentiators (what we lead with)

1. **Metric scale** — metric geometry + real poses → we can measure the room (most monocular pipelines can't), which is the prerequisite for the embodied stage.
2. **Multi-method consensus fusion** — run PGSR / DN-Splatter / MonoSDF and fuse them; one method's blind spot is another's strength.
3. **Rigorous GT-mesh benchmark** — Chamfer / F-score vs ground-truth meshes, not screenshots.
4. **Embodied world** — carry the metric mesh into a robot-explorable Habitat/Genesis environment.

---

## 4. Key technical gotchas (so we don't re-learn them)

- **GPU access:** UCL `trailbreaker` via 2-hop sshpass through `knuckles`; use project disk `/cs/student/projects3/2023/dkozlov` (home is over-quota); redirect TORCH_HOME/HF_HOME/pip/conda caches there.
- **torch/CUDA:** driver 12.6 → install torch `+cu126` (default mapanything pull was cu130 → "driver too old").
- **System CUDA is runtime-only** (no nvcc) → gsplat can't JIT-compile. The MapAnything env's conda-meta got corrupted by a failed cuda-toolkit install (still runs, but no `conda install`). **Solution: separate `splat` env (py3.10/torch2.2+cu121) with the PREBUILT gsplat wheel — no compile.**
- **TSDF (planned):** Open3D `ScalableTSDFVolume`, `depth_scale=1.0` (depth already metric), pass `inv(cam2world)`, real intrinsics; pure pip wheel (no compile, pin py≤3.12).
- **Pose A/B (planned):** `evo` (TUM format, `evo_ape -va`, `--correct_scale` only if scale uncertain). Skip ORB-SLAM3 (build pain); we already have two metric pose sources (ARKit + MapAnything).

---

## 4b. Public datasets to demo the pipeline at its best (researched 2026-05-27)

Our own capture is the limiter (low-res, blank walls, wall-pan → blobs). Clean public scenes fix all three (sharp, textured, 360 coverage → orbit looks good).

- **#1 hero (looks best, easiest, license-safe): Mip-NeRF 360 indoor — `room`/`bonsai`/`counter`/`kitchen`.** CC BY-SA 3.0. 1–1.6 MP, true 360 orbit, ships COLMAP poses+intrinsics (→ enables our pose-free vs pose-assisted A/B). Plugs straight into MapAnything's `demo_inference_on_colmap_outputs.py` (pose-assisted) / images-only (pose-free). NOT metric (COLMAP scale) — keep metric story for our ARKit/ScanNet++.
  `curl -O http://storage.googleapis.com/gresearch/refraw360/360_v2.zip` (~12 GB; has room/kitchen/counter/bonsai/bicycle/garden/stump)
- **#2 easiest phone-native: Nerfstudio sample data** (poses in transforms.json): `ns-download-data nerfstudio --capture-name kitchen`.
- **#3 metrics + GT (real phone video of a room): ScanNet++ v2** — laser-scan GT mesh + iPhone RGB-D video + DSLR + poses → best for TSDF-vs-GT + the A/B. CAVEAT: account + download-token + **non-commercial/academic gate** — the USER must register (I can't); fine for an internship code submission.
- **Replica** — synthetic GT mesh; render a flawless trajectory + the mesh IS the GT (no real-phone story).

Plan: **hero demo on Mip-NeRF 360 `room`/`bonsai`** (gorgeous, orbit-clean, poses for A/B), keep our **ARKit phone capture** as the metric + camera-systems story, optional **ScanNet++** for TSDF-vs-laser-GT metrics.

## 4c. PIPELINE VALIDATED on clean data (2026-05-27)

Ran the **exact same pipeline** on Mip-NeRF 360 `room` (80 of 311 views, `images_4`):
MapAnything pose-free → `recon.ply` (1.5M pts) + `cameras.json` → gsplat densification (100k→**1.09M** gaussians, 7000 steps) → `runs/mip360_room/splat.ply` (74 MB) + interactive viewer `viewer/splat_mip.html`.
**Result: sharp, photorealistic, coherent room (rug/floor/table/plant/walls), good from all angles** — night-and-day vs our own phone capture. **Confirms the bottleneck was the input, not the pipeline.** Generalization needed only a small adapter: `geometry/recon_folder.py` (pose-free on any image folder → ply + cameras.json) + `splat.py --cameras-json` (train on MapAnything's estimated cameras). Loads in mkkellogg viewer with 0 console errors.

This Mip-NeRF 360 result is the **hero demo** for the submission (CC BY-SA 3.0, attribute). Keep our own ARKit clip as the honest "real phone + metric + camera-systems" companion.

## 4d. REFERENCE-QUALITY splat achieved (2026-05-27)

Our hand-rolled gsplat trainer was the quality bottleneck (blurry/floaters). Switched to the **official gsplat `simple_trainer.py` (v1.4.0)** with **MCMC strategy** on Mip-NeRF 360 `room` (images_2, COLMAP poses, SH3, 30k steps, cap_max 1M):
**PSNR 32.3 / SSIM 0.93 / LPIPS 0.15, 1M gaussians** — at/above benchmark. Pure renders at captured viewpoints are photorealistic (crisp floor, readable objects). Setup: clone gsplat repo @v1.4.0 (examples not in wheel), **patch out `fused_ssim`** (needs nvcc) -> torchmetrics SSIM, install deps (rmbrualla pycolmap, viser, nerfview==0.0.2, lpips, etc.), train `mcmc --disable_viewer --data_factor 2 --strategy.cap_max 1000000`. Checkpoint at results/room_mcmc/ckpts/ckpt_29999. Convert .pt->.ply via plyfile (shN.permute(0,2,1) for channel-major f_rest).

KEY LESSON: reconstruction quality is now solved (use official trainer + MCMC). Remaining "bad" visuals are CAMERA-PATH artifacts: trajectory renders (ellipse/interp) wander between captured cameras into dark/under-observed spots; free-orbit hits coverage gaps. Fix = render along the actual captured camera path / constrain demo orbit to the captured band. Aggressive ply pruning (opacity>0.05 + 99.5pct scale + aniso<10 + bbox) over-cut to 143k/1M — too harsh; use the FULL 1M ply or gentle pruning.

## 4e. REAL2SIM LOOP WORKING — agent explores reconstructed room (2026-05-27)

Full pipeline implemented end-to-end: reconstruction -> mesh -> Habitat navmesh -> navigation agent -> first-person rollout video.
- `geometry/mesh.py` (Open3D Poisson: point cloud -> mesh, largest-component, decimate). NOTE Open3D's GLB *reader* is broken (ASSIMP buffer error) — read the OBJ instead.
- `scripts/align_mesh.py` — MapAnything world frame is arbitrarily oriented -> Habitat Recast navmesh FAILS ("Could not build Detour navmesh"). Fix: estimate gravity-up from mean camera up-vector (-c2w[:3,1]) in cameras.json, rotate mesh so up->+Y, drop floor to y~0. After this navmesh builds.
- `scripts/habitat_explore.py` (habitat-sim 0.3.3, `habitat` env py3.9 headless): load GLB scene, recompute_navmesh (agent radius/height/max_climb), GreedyGeodesicFollower to random navigable goals, render RGB -> mp4 (imageio). Ran: navigable_area 1.732, goals_reached 6/6, 126 frames -> runs/mip360_room/explore.mp4.
- CAVEAT: agent view renders the ROUGH HOLEY MESH (not the splat) -> looks rough/dark. Walkable area small (holey mesh). Visual fixes (documented): (a) cleaner mesh via 2DGS/SuGaR/TSDF; (b) **Habitat-GS** (github zju3dv/habitat-gs) to render the photorealistic SPLAT as the agent's camera while mesh handles collision — the "wow" version, more setup (custom CUDA build).
- This is the capstone: reframes submission from "I reconstructed a room" to "I built a working real2sim pipeline where an agent explores a phone-scanned room." Honest improvement axes documented.

## 4f. FULL real2sim pipeline WORKING — autonomous agent + photoreal render (2026-05-27)

End-to-end: phone-scan-style reconstruction -> splat -> mesh -> navmesh -> navigating agent -> photoreal first-person video. `runs/mip360_room/explore_photoreal.mp4` (201 frames, smooth, photoreal: sofa/bookshelf/plant). DECOUPLED design (no Habitat-GS CUDA build): Habitat does nav, gsplat renders pixels, only camera poses pass between them.

Pipeline + the bugs solved (all in `scripts/`):
1. `gsplat_tsdf_mesh.py` — render gsplat depth (render_mode RGB+ED) at COLMAP poses, Open3D TSDF -> mesh IN SPLAT FRAME (co-registered). `--poisson` reconstructs from fused points to FILL FLOOR GAPS (continuous navmesh surface). Emits `scene.json` with `M_s2h` (source->Habitat transform) + sanity render (confirmed frame consistency — photoreal living room).
2. `export_habitat_glb.py` — pre-rotate mesh Rx(+90) to CANCEL Habitat's GLB-import rotation (Open3D GLBs import rotated; OBJ stays Y-up but OBJ stages segfault recompute_navmesh). So Habitat operates in the aligned frame -> M_s2h maps poses back cleanly.
3. `habitat_record_path.py` — Habitat navmesh + greedy follower, records agent cam->world poses + intrinsics. **KEY navmesh fixes: (a) GLB import re-orients (use Rx+90 GLB); (b) `fill_holes` introduced toxic geometry -> SEGFAULT (drop it); (c) TSDF floor too gappy -> recompute returns False (use --poisson); (d) the real unlock: `agent_max_climb=1.0` + small agent (r0.05,h0.3) + cell_size 0.05 for the COLMAP-NORMALIZED scale -> navigable_area 7.6 (climb 0.3 was rejecting the floor). Guard pf.navigable_area access or it segfaults on failed builds.**
4. `gsplat_render_path.py` — map each Habitat pose -> splat frame (`Minv @ cam2world_H @ diag(1,-1,-1,1)`), render splat. K from Habitat hfov. Crop frames to EVEN dims (h264). 
5. `gsplat_render_traj.py` — alternative: render splat along the capture trajectory (smooth photoreal tour, `tour.mp4`).

CAVEAT: peripheral blur when the agent leaves the captured viewpoint manifold (gets close to walls) — inherent splat coverage limit; mitigate by constraining the agent to the central region.

## 5. Recommended next steps (highest leverage first)

1. **TSDF mesh** from MapAnything depth+poses (Open3D) — fixes blobs, differentiator, low risk.
2. **Pose A/B metrics** with `evo` (ARKit vs pose-free vs pose-assisted) + metric scale check vs the real room.
3. **Package** well: results-first README (hero walkthrough GIF + hosted viewer link + metrics table), explicit "design decisions & tradeoffs", one-command run, sample input.
4. (Optional) **Re-capture or use native 1920×1440 photos** for a sharper result; (optional) 3D semantics via SAM2+CLIP.
