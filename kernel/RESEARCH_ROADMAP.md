# Toward world-class reconstruction: separation fix + the crack-level north star

Research synthesis (4 parallel SOTA sweeps, 2024-2026) + experimental roadmap.
Goal: fix object separation on real data, and chart the realistic path to
reconstruction fine enough to detect sub-mm wall cracks.

---

## The one insight that reframes everything

**Detecting a sub-mm crack is not a geometry problem — it is a photometric problem.**

Multi-view geometry (what every method does, including ours) has a hard resolution
floor set by triangulation: depth error Δz ∝ z²/(f·b). At a ~2 m wall standoff that
floor is **multiple millimetres**, and worse on a textureless wall (no disparity
signal). So:

- The SOTA object-scale methods (Neuralangelo 0.61 mm, PGSR 0.52 mm, MILo 0.68 mm,
  2DGS 0.80 mm on DTU) hit ~0.5-0.8 mm *whole-object mean* error **only** in lab
  conditions (dozens of high-res views, ~0.5 m range, diffuse light, textured object).
  None demonstrate reliable single-feature sub-mm recovery, and none work on a flat
  wall at room distance.
- Our kernel (~0.9-1.3 cm) is already at the **casual-phone floor**. The binding
  constraint is multi-view depth-noise averaging, **not** our voxel size — shrinking
  voxels below the depth-noise floor buys nothing.
- A sub-mm crack's depth signal sits **below** that floor at any realistic wall
  standoff. No amount of geometric refinement recovers it; the information is not in
  the disparities.

What *does* carry sub-mm relief is **shading / surface normals**. A crack casts
shading that shifts with light direction; that signal is per-pixel and independent of
the triangulation baseline. Photometric stereo / multi-view photometric stereo (MVPS)
reaches **0.15-0.2 mm** true-geometry detail (SuperNormal, NPLMV-PS); RTI in art
conservation resolves hundreds-of-microns crack relief from multi-light capture.

**Crucially, our coarse base surface is the enabler, not a competitor.** Shape-from-
shading is unusable alone because of the bas-relief ambiguity (absolute flatness is
unrecoverable from shading); the coarse multi-view surface supplies exactly the
low-frequency shape that resolves it. So the world-class system is:

> **coarse metric base (our kernel)  +  high-frequency photometric normals  +
> discontinuity-preserving integration → displacement/normal detail.**

Honest boundary: from a **single casual photo**, predicted normals give a plausible
but *un-certifiable* detail illusion (bas-relief + albedo-vs-shading ambiguity, and
monocular normal nets are band-limited at crack scale). Certifying a crack as true
geometry (real groove vs painted line) **requires multiple light directions** — e.g.
flash-on/flash-off or a moving phone LED. That is a capture-protocol change, not a
kernel tweak.

---

## Part A — Fix object separation (actionable now)

The research verdict: our per-voxel label-voting kernel is architecturally sound — it
is how 2024-26 SOTA works under the hood. **The fix is the mask source and the
association, not the kernel.** Naive per-frame SAM masks over-segment and flicker,
which is what breaks voting.

**Tier 0 (minimal, do first — works with our existing voting):**
1. Replace per-frame masks with **SAM 2 video-tracked masks** (seed objects, propagate
   stable IDs across the sweep). Removes most flicker.
2. Add a **Hungarian association** step between each frame's track IDs and our global
   voxel-vote IDs (the Panoptic Lifting recipe), so a re-seen object reuses its ID.
3. Gate votes: require a **minimum vote margin** before committing a voxel, + a local
   majority filter to kill speckle.

**Tier 1 (robust):**
4. Pool votes over **geometric superpoints** (normal-based graph-cut, SAI3D-style)
   instead of independent voxels — a superpoint won't straddle a chair-leg/floor
   discontinuity, so contact patches separate cleanly.
5. **View-consensus merging** (MaskClustering): merge two regions only if multiple
   *other* views agree they co-occur.
6. Explicit **shell handling**: floor/walls/ceiling = large planar surfaces (plane-fit
   or semantic) + any low-confidence voxel → forced into the room shell, so object
   masks can't bleed onto structure.

**Tier 2 (labels included, less engineering):** SAM 3 (automatic, named, tracked
concept masks, Dec 2025) as the mask source, or run **MaskClustering / Open3DIS** on
the reconstructed cloud directly.

**Metrics:** class-agnostic instance AP (AP/AP50/AP25) for separation quality, and
**scene-level PQ** for multi-view ID consistency. Datasets: ScanNet++ (real, has an
instance benchmark), Replica (prototyping).

Implementation note: SAM 2 needs the GPU box; the kernel-side changes (Hungarian
association, confidence gating, superpoint pooling) can be written now and slotted into
the existing voting path.

---

## Part B — The crack-level reconstruction system

### B1. Geometry track (push the base as far as it honestly goes)
- Add **multi-view + planar regularization to the surfel stage before Poisson** — this
  is exactly the 2DGS→PGSR jump (0.80→0.52 mm DTU). Highest-leverage upgrade to our
  existing surfel+Poisson path.
- Consider a **hash-grid SDF with numerical gradients + coarse-to-fine** (Neuralangelo
  recipe) or a **Gaussian→SDF hybrid** (SurfaceSplat/3DGSR) for coherent refinable
  detail beyond TSDF discretization.
- **Mesh-in-the-loop (MILo)** instead of post-hoc Poisson to stop thin-feature erosion.
- But: this track tops out at mm-class macro-geometry. It will **not** reach sub-mm.

### B2. Photometric track (the only route to sub-mm — build this)
- **Normals:** StableNormal (best sharpness/stability, 13.7° MAE) for monocular; or
  near-light / multi-view photometric stereo (SuperNormal, NPLMV-PS) for the
  certified version under multi-light capture.
- **Integration:** Bilateral Normal Integration (BiNI, ECCV 2022) — preserves the
  one-sided depth discontinuity that *is* a crack, unlike Poisson normal integration
  which rounds it off.
- **Fusion onto the base:** UV-parameterise the base mesh → high-pass the predicted
  normals → graft only the high-frequency residual onto the base's own low-frequency
  normals (base owns absolute shape, net owns detail) → BiNI → displacement + normal
  map. Precedent: Yu CVPR'13 shading refinement, Robertini'16 per-vertex displacement.
- **Certification (real groove vs paint):** keep a crack as displacement only if it
  survives albedo/shading separation and shifts correctly under changing light —
  which requires multi-light capture (flash-on/off). Single-shot output is labelled
  "appearance-level enhancement, not certified geometry."

---

## Part C — New metrics (mean Chamfer is blind to cracks)

A crack is a vanishing fraction of surface area, so whole-scene mean Chamfer averages
it away. Adopt, in priority order:

- **P0 Crack-localised metrics (north star):** in a ±2-3 mm tube around the GT crack
  centreline, report **crack-detection F-score**, **width-MAE**, **depth-MAE**. Report
  the minimum crack width at which F-score ≥ 0.5 — that single number is our "crack
  resolution."
- **P1 F-score swept to sub-mm thresholds:** the threshold where F-score collapses is
  the effective resolution. Cheapest upgrade to what we already compute.
- **P2 Normal MAE / normal-consistency:** high-frequency geometry lives in normals;
  the standard detail proxy (used by Neuralangelo, 2DGS, all photometric-stereo evals).
- **P3 Detail-band (high-pass) Chamfer:** Laplacian-smooth both meshes, subtract,
  Chamfer the high-pass residual (or SAUCD spectral metric, CVPR 2024).
- **P4 PSD comparison** on planar patches (where does detail die in frequency).

### Ground truth
No public dataset has posed-RGB-D input + sub-mm crack GT together — a genuine gap.
Build one: real cracked wall patches, GT from a **structured-light / chromatic-confocal
profilometer** (sub-mm to µm, validated against caliper width), our posed RGB-D + RGB
of the same patches, crack centreline/width/depth annotations. Borrow eval machinery
from civil-engineering crack inspection (CrackSeg9k, structured-light 0.16 mm width
MAE) and µCT crack GT (VoroCrack3d, 2.8-106 µm/voxel).

---

## Part D — Falsifiable hypotheses (the experiments that decide the architecture)

Capture per wall patch with profilometer GT; vary one factor at a time
(resolution, view count, baseline, lighting).

- **H1 — geometric floor.** Crack width/depth error plateaus at ~the depth-noise σ
  (mm-scale) and does **not** improve with finer voxels or higher image resolution
  beyond that. *Falsified if voxel/resolution sweeps keep improving sub-mm.*
- **H2 — Poisson destroys detail.** Raw oriented surfels score strictly better on
  crack high-pass-Chamfer and crack F-score than the Poisson mesh; Poisson only helps
  whole-scene Chamfer. *Falsified if Poisson ties/wins on crack metrics.*
- **H3 — photometric beats the floor.** Multi-light photometric normals cut crack
  width/normal error **below** the H1 geometric floor, and lighting variation is the
  only factor that pushes F-score≥0.5 width below ~1 mm. *Falsified if photometric
  adds nothing past the floor.*
- **H4 — metric sensitivity.** Whole-scene mean Chamfer stays flat across all sweeps
  while crack-localised metrics move sharply — proving the metric change (not just the
  pipeline) is what surfaces detail. *Falsified if mean Chamfer tracks crack metrics.*
- **H5 — base resolves bas-relief.** Shape-from-shading anchored on our coarse base
  recovers correct absolute crack depth, while unanchored SfS does not (bas-relief).
  *Falsified if anchoring doesn't improve absolute depth error.*
- **H6 (separation) — association, not voting.** SAM2-tracking + Hungarian association
  raises scene-PQ far more than any change to the voting rule. *Falsified if voting
  tweaks alone close the gap.*

### Ablations
A1 voxel sweep (8/4/2/1 mm) · A2 Poisson on/off (vs raw surfels) · A3 depth-noise
injection (detail floor vs σ) · A4 view-count sweep (detail vs noise suppression) ·
A5 photometric add-on (multi-light normals) · A6 mask source (per-frame SAM vs SAM2
tracking) · A7 vote pooling (per-voxel vs superpoint).

---

## Part E — Build order on top of our kernel

1. **Separation Tier 0** (SAM2 tracking + Hungarian + vote gating) — fixes the headline
   request, measurable in scene-PQ / instance AP on Replica then ScanNet++.
2. **Crack metric harness** (P0-P3) + a small profilometer/structured-light GT test set
   — without this, nothing about "cracks" is measurable.
3. **H1/H2/H4 ablations** — establish our geometric floor and prove the metric change.
   Expected: confirms geometry alone floors at ~mm, Poisson erodes detail.
4. **Photometric branch (B2)** — StableNormal normals + BiNI integration + base-anchored
   high-pass fusion. Run H3/H5. This is the genuinely novel, world-class-aiming piece.
5. **Geometry upgrades (B1)** — planar/multi-view surfel reg, then SDF/MILo — to push
   the base as far as honest, feeding the photometric branch a better anchor.

### Honest framing for the north star
"Perfect quality, detect the smallest cracks" is achievable **as true geometry only
under controlled multi-light capture** (MVPS regime, 0.15-0.2 mm demonstrated). From
casual phone video it is achievable **as plausible appearance-level detail**, not
certified geometry, bounded by the bas-relief and albedo-shading ambiguities. The
defensible world-class contribution is the **base + photometric-detail fusion** with a
**crack-localised metric and a profilometer GT benchmark that does not yet exist
publicly** — that benchmark is itself a publishable artifact.

## Key references
Separation: SAM2 2408.00714 · SAM3 2511.16719 · Panoptic Lifting 2212.09802 ·
SAI3D 2312.11557 · MaskClustering 2401.07745 · Open3DIS 2312.10671 · AutoSeg3D
2512.07599 · Gaussian Grouping 2312.00732.
Geometry: Neuralangelo 2306.03092 · 2DGS 2403.17888 · PGSR 2406.06521 · RaDe-GS
2406.01467 · MILo 2506.24096 · SurfaceSplat 2507.15602.
Photometric: SuperNormal 2312.04803 · NPLMV-PS 2405.12057 · LUCES-MV 2412.16737 ·
StableNormal 2406.16864 · BiNI (ECCV 2022) · bas-relief ambiguity (Belhumeur-Kriegman-
Yuille IJCV 1999) · Yu shading refinement CVPR 2013 · Robertini 1602.02023.
Metrics/GT: SAUCD 2403.01619 · DiLiGenT-Π (ICCV 2023) · VoroCrack3d (PMC11109345) ·
CrackSeg9k 2208.13054 · FineRecon 2304.01480.
