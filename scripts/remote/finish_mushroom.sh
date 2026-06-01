#!/bin/bash
# Post-training orchestration for MuSHRoom coffee_room:
#  1) fuse PGSR (backbone) + DN (gap-fill) -> consensus mesh
#  2) eval PGSR / DN / consensus vs Faro gt_mesh.ply (mushroom vis-cull protocol)
#  3) render an interior fly-through of the chosen recon mesh + encode mp4
# Meshes are already in the iphone/COLMAP metric frame (same as PGSR's COLMAP);
# the mushroom eval auto-aligns GT via icp_iphone.json.
set -x
ROOT=/cs/student/projects3/2023/dkozlov
DNENV=$ROOT/conda-envs/dnsplat
PGENV=$ROOT/conda-envs/pgsr
SPLAT=$ROOT/conda-envs/splat/bin/python
W=$ROOT/work
CR=$ROOT/datasets/mushroom/room_datasets/coffee_room
LC=$CR/iphone/long_capture
LOCK=$ROOT/tmp/gpu.lock
MLOCK=$ROOT/tmp/mesh.lock
export TMPDIR=$ROOT/tmp PYOPENGL_PLATFORM=egl
export TORCH_HOME=$ROOT/torch-home HF_HOME=$ROOT/hf-home CUDA_VISIBLE_DEVICES=0

# locate the trained meshes
PGSR=$(ls -t $W/out/pgsr_mushroom_coffee/*/mesh/tsdf*.ply $W/out/pgsr_mushroom_coffee/mesh/tsdf*.ply 2>/dev/null | head -1)
DN=$(ls -t $W/out/dn_mushroom_coffee/mesh/*.ply 2>/dev/null | head -1)
echo "PGSR_MESH=$PGSR"
echo "DN_MESH=$DN"

OUTF=$W/out/mushroom_coffee_fused
mkdir -p $OUTF

# ---- 1) consensus fusion (PGSR backbone, DN gap-fill, identity frame) ----
if [ -n "$PGSR" ] && [ -n "$DN" ]; then
  $SPLAT $W/fuse_consensus.py --pgsr "$PGSR" --dn "$DN" --identity-dn \
    --out $OUTF/consensus.ply --tau-dn 0.05 --depth 11 --trim 0.02 2>&1 | tail -6
  echo "FUSE_RC=$?"
fi

# ---- 2) eval each mesh vs Faro GT ----
cd $ROOT/dn-splatter-src
evalmesh () {
  local PRED=$1; local NAME=$2
  [ -z "$PRED" ] && { echo "SKIP_EVAL_$NAME (no mesh)"; return; }
  local OD=$W/eval/mushroom_coffee_${NAME}
  mkdir -p $OD
  flock $MLOCK -c "$DNENV/bin/python dn_splatter/eval/eval_mesh_mushroom_vis_cull.py \
    --gt-mesh-path $CR --pred-mesh-path $PRED --device iphone \
    --output $OD --output-same-as-pred-mesh False \
    --rename-output-file mushroom_coffee_${NAME}_metrics.json"
  echo "EVAL_RC_${NAME}=$?"
  echo "--- metrics $NAME ---"; cat $OD/mushroom_coffee_${NAME}_metrics.json; echo
}
evalmesh "$PGSR" pgsr
evalmesh "$DN" dn
evalmesh "$OUTF/consensus.ply" consensus

# ---- 3) fly-through render of PGSR mesh (the textured backbone) ----
FRAMES=$W/out/mushroom_coffee_fly
mkdir -p $FRAMES
$DNENV/bin/python $W/render_mesh_mushroom.py "$PGSR" $LC $FRAMES/fly --stride 4 --scale 1.0 2>&1 | tail -3
echo "RENDER_RC=$?"
# encode mp4 (ffmpeg from dnsplat env or system)
FF=$(which ffmpeg 2>/dev/null || echo $DNENV/bin/ffmpeg)
$FF -y -framerate 20 -i $FRAMES/fly_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 20 \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" $W/out/mushroom_coffee_flythrough.mp4 2>&1 | tail -3
echo "FFMPEG_RC=$?"
ls -la $W/out/mushroom_coffee_flythrough.mp4
echo "FINISH_MUSHROOM_DONE"
