#!/bin/bash
# DN-backbone consensus: swap the fusion so DN-Splatter (the stronger method) is
# the backbone and PGSR is the gap-filler (the reverse of the default). Tests
# whether starting from the better geometry + borrowing PGSR only where DN
# missed can beat the best single method. No retraining -- reuses existing
# per-scene meshes; just re-fuse + re-score.
ROOT=/cs/student/projects3/2023/dkozlov
W=$ROOT/work
SPLAT=$ROOT/conda-envs/splat/bin/python
exec > $W/dn_backbone.log 2>&1
echo "=== DN-BACKBONE START $(date) ==="
for s in room0 room1 room2 office0 office1; do
  DN=$W/out/dn_${s}/mesh/Open3dTSDFfusion_mesh.ply
  PGSR=$W/out/pgsr_${s}/mesh/tsdf_fusion_post.ply
  OUT=$W/out/consensusB_${s}
  mkdir -p $OUT
  echo "--- fuse DN-backbone $s ---"
  # --pgsr = backbone (DN here); --dn = gap-fill (PGSR here); identity frame (Replica)
  $SPLAT $W/fuse_consensus.py --pgsr $DN --dn $PGSR --identity-dn \
    --out $OUT/consensusB.ply --tau-dn 0.05 --depth 11 --trim 0.02 2>&1 | tail -2
  bash $W/eval_scene.sh $s $OUT/consensusB.ply consensusB 2>&1 | grep -aE "EVAL_RC|F-score|metrics ===" | tail -2
done
echo "=== DN_BACKBONE DONE $(date) ==="
