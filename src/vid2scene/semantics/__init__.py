"""Stage 4 — geometry-coherent 3D semantics (PLANNED, GPU).

Core (real code, ships):   per-frame SAM2 + Grounding DINO masks -> project
   into 3D via depth+pose -> multi-view weighted voting -> per-point labels.
   Coherence is mechanical: labels come from the same depth that made the
   geometry. A k-NN/CRF smoothing pass enforces spatial consistency.

Upgrade (real code):       Gaussian Grouping (Apache) — identities baked into
   the Gaussians, optimised with a 3D consistency term.

Future work (paper-only as of 2026-05): SegSplat / SemanticSplat — feed-forward
   joint geometry+semantics. Re-check for a code release before submission.

Planned interface:
    def label(recon, frames_dir, classes: list[str]) -> LabeledScene
"""
