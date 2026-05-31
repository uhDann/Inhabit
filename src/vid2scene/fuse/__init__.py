"""Multi-method consensus fusion — the original contribution of vid2scene.

Run several reconstructors on the same video (PGSR, DN-Splatter, MonoSDF), then
fuse their meshes into one. The fusion is deliberately the *safe* kind from the
multi-reconstruction literature: a trusted backbone + gap-fill from the donor
*only where the backbone has holes* (visibility/selection-gated), NOT blind
volumetric averaging — which would manufacture doubled walls and smear away
detail when the methods disagree geometrically.

See `consensus.fuse_consensus`. The GT-mesh benchmark (../benchmark) quantifies
when this helps: it trades a little Accuracy for more Completion, a net win on
gappy / cluttered scenes (validated on Replica office1; +45% over the PGSR
backbone on room2).
"""

from .consensus import ConsensusConfig, fuse_consensus

__all__ = ["ConsensusConfig", "fuse_consensus"]
