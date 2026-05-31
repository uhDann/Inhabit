#!/bin/bash
# Re-extract the MonoSDF tight watertight mesh from the epoch-200 checkpoint
# at marching-cube resolution 512 (the training-time plot OOM'd at the same
# resolution because the optimiser state + activations + grid all coexisted;
# with training over we have all 16 GB free, so it should fit).
set -x
ROOT=/cs/student/projects3/2023/dkozlov
SRC=$ROOT/monosdf-src
ENV=$ROOT/conda-envs/monosdf
LOCK=$ROOT/tmp/gpu.lock
TS=2026_05_29_13_05_24
CKPT_EPOCH=200
RES=${RES:-512}
CKPT_PATH=$SRC/exps/room_grids_tight_1/$TS/checkpoints/ModelParameters/$CKPT_EPOCH.pth

export CUDA_VISIBLE_DEVICES=0
export PIP_CACHE_DIR=$ROOT/pip-cache TORCH_HOME=$ROOT/torch-home HF_HOME=$ROOT/hf-home TMPDIR=$ROOT/tmp
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SRC/code"
# eval.py's --checkpoint takes a full FILE PATH (it calls torch.load on it
# directly, not an epoch number). The conf's expname identifies the exp dir.
flock "$LOCK" -c "$ENV/bin/python evaluation/eval.py \
  --conf confs/room_grids_tight.conf \
  --checkpoint $CKPT_PATH \
  --resolution $RES \
  --world_space"
echo "EVAL_RC=$?"
echo "EXTRACT_DONE"
