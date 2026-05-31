#!/bin/bash
# Parallel multi-scene Replica benchmark on ONE 16GB GPU.
# Memory-safe design: two concurrent training STREAMS (one PGSR ~6GB + one DN
# ~3GB = ~9-11GB), each running its scenes sequentially. This keeps <=2 heavy
# trainings on the card at any time (running two coupled drive_scene.sh would be
# ~17GB -> OOM). Mesh-extraction + eval GPU spikes are serialized via mesh.lock
# inside the per-scene scripts. 15k iters (set in launch_*_scene.sh).
ROOT=/cs/student/projects3/2023/dkozlov
W=$ROOT/work
SPLAT=$ROOT/conda-envs/splat/bin/python
SCENES="${SCENES:-room1 room2 office0 office1}"
exec > $W/bench_parallel.log 2>&1
echo "=== BENCH PARALLEL START $(date) :: scenes=[$SCENES] ==="

# 1. CPU prep (quick): COLMAP model for PGSR + eval_transforms for culling
for s in $SCENES; do
  $SPLAT $W/build_colmap.py $s > $W/colmap_$s.log 2>&1 && echo "colmap $s OK" || echo "colmap $s FAIL"
  $SPLAT $W/make_eval_transforms.py $s > $W/mket_$s.log 2>&1 && echo "transforms $s OK" || echo "transforms $s FAIL"
done

# 2. Two concurrent training streams (<=2 trainings on the GPU at once)
( for s in $SCENES; do echo "[PGSR] start $s $(date +%H:%M:%S)"; \
    bash $W/launch_pgsr_scene.sh $s > $W/pgsr_$s.log 2>&1; \
    echo "[PGSR] done $s $(date +%H:%M:%S)"; done ) &
PSTREAM=$!
( for s in $SCENES; do echo "[DN] start $s $(date +%H:%M:%S)"; \
    bash $W/launch_dn_scene.sh $s > $W/dn_$s.log 2>&1; \
    echo "[DN] done $s $(date +%H:%M:%S)"; done ) &
DSTREAM=$!
wait $PSTREAM; echo "=== PGSR STREAM DONE $(date) ==="
wait $DSTREAM; echo "=== DN STREAM DONE $(date) ==="

# 3. Finalize each scene: mesh -> consensus -> eval (mesh.lock serializes spikes)
for s in $SCENES; do
  echo "--- finalize $s $(date +%H:%M:%S) ---"
  bash $W/extract_pgsr_mesh.sh $s > $W/pgsr_mesh_$s.log 2>&1
  bash $W/extract_dn_mesh_scene.sh $s > $W/dn_mesh_$s.log 2>&1
  mkdir -p $W/out/consensus_$s
  $SPLAT $W/fuse_consensus.py \
    --pgsr $W/out/pgsr_$s/mesh/tsdf_fusion_post.ply \
    --dn $W/out/dn_$s/mesh/Open3dTSDFfusion_mesh.ply --identity-dn \
    --out $W/out/consensus_$s/consensus.ply \
    --tau-dn 0.05 --depth 11 --trim 0.02 > $W/fuse_$s.log 2>&1
  bash $W/eval_scene.sh $s $W/out/pgsr_$s/mesh/tsdf_fusion_post.ply pgsr > $W/eval_${s}_pgsr.log 2>&1 || true
  bash $W/eval_scene.sh $s $W/out/dn_$s/mesh/Open3dTSDFfusion_mesh.ply dn > $W/eval_${s}_dn.log 2>&1 || true
  bash $W/eval_scene.sh $s $W/out/consensus_$s/consensus.ply consensus > $W/eval_${s}_consensus.log 2>&1 || true
  echo "FINALIZED $s $(date +%H:%M:%S)"
done
echo "=== BENCH PARALLEL DONE $(date) ==="
