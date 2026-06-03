# Photoreal mesh from phone video: "confusable with the real room", on true geometry

Research synthesis (2 SOTA sweeps, 2024-2026) + novel experiment program.
North star: a real 3D triangle MESH that renders as photorealistically as a Gaussian
splat — a person looking at it can't tell it from the actual room — while remaining
true, editable, physics-ready geometry (not view-dependent splat blobs).

---

## Why this is achievable (and why it plays to OUR strengths)

The mesh-vs-splat visual gap is **real but small and well-characterised: ~0.7-1.5 dB
PSNR, +0.02-0.04 LPIPS** on standard benchmarks. And the decisive finding:

**The gap is NOT geometry. It is appearance.** Ranked by contribution:
1. **Soft alpha blending / anti-aliasing** at silhouettes and thin structures (a splat's
   continuous alpha falloff gives free AA; a hard triangle has binary coverage).
2. **View-dependent appearance** (specular highlights, glossy floors) — a flat-textured
   mesh looks plastic; per-texel view-dependent color is what makes it look real.
3. **Texture resolution.**
4. **Geometry accuracy — the SMALLEST factor, and the one we already solved (~1 cm).**

Splats win on *appearance*, not 3D — splat geometry is actually noisy/hollow, which is
why SuGaR/2DGS/GS2Mesh exist to extract meshes from them. So our accurate metric mesh
is an **asset, not a liability**. The proof: a co-optimised textured mesh **beats its
source splat** once geometry is accurate (Texture-Guided Gaussian-Mesh Joint Opt,
2511.03950: mesh 26.21 vs splat 23.82 PSNR on DTU). We start from accurate geometry, so
we skip the hard part everyone else fights.

**Conclusion: a mesh with per-texel view-dependent appearance + anti-aliasing can match
or beat a splat on novel-view realism — and it's a true editable physics-ready mesh.**
Nobody has cleanly packaged "phone video → photoreal, physics-ready mesh that rivals
splats." That is the boundary to push.

---

## The buildable pipeline (rasterizer-native, runs in any engine / on-device)

1. **Geometry = our kernel.** Keep the metric surfel + Poisson mesh (our asset).
2. **Sharp diffuse atlas** via MRF multi-view texturing ("Let There Be Color!",
   Waechter 2014): per-face best-view selection (sharpness, fronto-parallel,
   unoccluded) + global + Poisson seam levelling for exposure/color harmonisation.
   Robust to phone-video blur/exposure.
3. **View-dependent appearance layer** (the core of the splat look): per-texel
   **diffuse + a few spherical-Gaussian specular lobes** (BakedSDF recipe, 2302.14859)
   — analytic, rasterizer-native, no per-frame CNN, no flicker.
4. **Differentiable fine-tune**: optimise the appearance (texture + SG specular) with
   **nvdiffrast** (2011.03277) under an **L1 + LPIPS** photometric loss against ALL
   input frames, geometry frozen. Closes most of the residual.
5. **Anti-aliasing**: MSAA / supersampling + nvdiffrast analytic AA (and optionally a
   thin SuGaR-style semi-transparent surfel shell only at silhouettes) to fight the
   splat's soft-edge advantage — the largest remaining gap.

Expected landing: **within ~0.3-1.0 dB / +0.01-0.02 LPIPS of a tuned 3DGS**, ahead on
geometry-stressed regions, as a true mesh.

---

## The north-star test: "can you tell it's a reconstruction?"

Mean reconstruction error is irrelevant here; perceived realism is the target.

**Protocol (held-out novel-view synthesis):** train/build on a subset of frames, render
from **held-out poses the system never saw**, compare to the real photos. Use the
NerfBaselines split (every 8th frame test).

**Objective metrics + "indistinguishable" anchors:**
- **LPIPS** (lead metric) — ≲ 0.10 perceptually very close; **≲ 0.05 effectively
  indistinguishable** side-by-side. Report **DreamSim** alongside (better human
  correlation for real-vs-rendered).
- **SSIM ≳ 0.95**, **PSNR ≳ 30 dB**, **FID** low (distributional realism).
- **vs splat:** mesh within **~0.01-0.02 LPIPS** of a tuned 3DGS on the same capture.

**The definitive claim — human 2AFC study:** show observers a real held-out photo vs our
same-pose render, forced choice, randomised. **Target: discrimination ≈ 50% (chance) =
confusable with the real room.** >65-70% means people can still tell. (Protocol:
"Assessing Photorealism of Rendered Objects," 2407.01767; 3DGS-IEval-15K.)

This benchmark — paired mesh-vs-splat-vs-real held-out NVS + a 2AFC confusion study on
phone-captured rooms — is itself a contribution; no clean public benchmark proves
"photoreal mesh confusable with the real room."

---

## Novel experiments + falsifiable hypotheses

- **E1 — The confusion benchmark (build first).** Held-out NVS of our mesh vs a tuned
  3DGS vs real photos, all metrics + 2AFC. Establishes the gap and the north-star metric.
- **E2 — View-dependent appearance.** Flat texture vs +spherical-Gaussian specular.
  **H1: VD appearance closes the majority of the LPIPS gap vs flat texture (≈1-3 dB on
  glossy indoor).**
- **E3 — Geometry-as-asset.** Same appearance pipeline on (a) a splat-derived SuGaR mesh
  vs (b) our accurate mesh. **H2: our accurate geometry + appearance BEATS the
  splat-derived mesh, and matches/beats the splat itself (replicating 2511.03950 at room
  scale).** This is the core thesis.
- **E4 — Differentiable appearance fine-tune.** nvdiffrast L1+LPIPS. **H3: closes to
  within ≤0.02 LPIPS of the splat.**
- **E5 — Anti-aliasing ablation.** MSAA/analytic-AA vs none. **H4: AA is the single
  largest remaining contributor to the mesh-splat gap (the soft-edge advantage).**
- **E6 — Appearance super-resolution.** SuperGaussian-style video-prior, input-anchored.
  **H5: input-constrained SR LOWERS held-out LPIPS (sharper AND consistent) without
  raising 2AFC discriminability; unconstrained per-view diffusion RAISES it (flicker).**
- **E7 — Relightable bet (boundary-pushing).** nvdiffrecmc + a material diffusion prior
  (MaterialFusion / IntrinsicAnything) → a mesh that RELIGHTS (splats can't). **H6:
  object-grade relighting is convincing (PSNR-L ~31-33); room-scale relighting under
  unknown indoor light is the open frontier — quantify exactly where it breaks.**

### The genuinely novel contribution to aim for
**Splat-appearance distillation onto an accurate mesh.** Train a 3DGS purely as an
*appearance teacher*; distill its view-dependent radiance into our mesh's per-texel
SG/neural appearance via differentiable rendering (the splat is the supervision target
from arbitrary synthesised views, not just the captured frames). Result: a mesh that
**inherits the splat's photorealism** while being true geometry — "best of both."
Combined with our metric scale + separable objects + physics-ready export, this is a
coherent, novel system: **phone video → photoreal, relightable-capable, physics-ready,
editable 3D mesh that is confusable with the real room.** No one has packaged this.

---

## Build order
1. **E1 confusion benchmark + a 3DGS baseline** on a phone-captured room (and Replica
   for paired GT). Without the metric, nothing here is provable.
2. **E2/E3:** view-dependent appearance baking on our mesh; prove geometry-as-asset
   (our mesh ≥ splat-derived mesh). The core thesis, fastest to a result.
3. **E4/E5:** nvdiffrast fine-tune + AA → reach within ~0.02 LPIPS of the splat.
4. **Distillation:** splat-teacher → mesh appearance (the novel piece) → push to 2AFC ~50%.
5. **E6/E7:** super-resolution (sharper-than-capture) and the relightable frontier.

Honest framing: parts 1-4 are a credible path to **splat-quality photoreal meshes from
phone video** (the achievable boundary-push, strongly supported by 2024-26 evidence).
The relightable room-scale mesh (E7) is the genuinely open frontier worth a real shot.

## Key references
Appearance: BakedSDF 2302.14859 · Neural Textures/Deferred 1904.12356 · MobileNeRF
2208.00277 · NeRF2Mesh 2303.02091 · nvdiffrast 2011.03277 · nvdiffrec 2111.12503 ·
nvdiffrecmc 2206.03380 · MeshLRM 2404.12385.
Splat→mesh: SuGaR 2311.12775 · 2DGS 2403.17888 · Texture-GS 2403.10050 · Joint
Gaussian-Mesh 2511.03950 · GS2Mesh 2404.01810.
Texturing: "Let There Be Color!" Waechter 2014 (mvs-texturing) · TexPainter 2406.18539.
Relight: NeRFactor 2106.01970 · TensoIR 2304.12461 · GS-IR 2311.16473 · Relightable-3DGS
2311.16043 · MaterialFusion 2409.15273 · IntrinsicAnything 2404.11593 · LightSwitch
2508.06494 · Stanford-ORB 2310.16044.
Eval: LPIPS 1801.03924 · DreamSim 2306.09344 · NerfBaselines 2406.17345 · 3DGS-IEval-15K
2506.14642 · photorealism user study 2407.01767.
Super-res: SuperGaussian 2406.00609 · DiSR-NeRF 2404.00874 · GaussianSR 2406.10111 ·
NeRF-SR 2112.01759.
