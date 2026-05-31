"""Quantitative GT-mesh benchmark — the rigor almost nobody in the field has.

We score each reconstruction (and the consensus) against Replica ground-truth
meshes with the standard MonoSDF/GO-Surf protocol: Accuracy, Completion,
Chamfer-L1, F-score@5cm, Normal-Consistency — after visibility-culling both GT
and prediction to the camera-observed region (the make-or-break step).

The culled eval itself runs on the GPU box (dn-splatter's eval_mesh_vis_cull.py
over all training poses); this module is the laptop-runnable collation layer
that turns the per-method metric JSONs into the comparison table.

See ../../docs/BENCHMARK.md for the full protocol + results.
"""

from .table import collate, METRICS

__all__ = ["collate", "METRICS"]
