#!/bin/bash
# Resume MonoSDF training with the tightened scene-bounding-sphere conf.
# Behind the GPU lockfile (forward-compat with other GPU jobs).
set -x
ROOT=/cs/student/projects3/2023/dkozlov
SRC=$ROOT/monosdf-src
ENV=$ROOT/conda-envs/monosdf
LOG=$ROOT/vid2scene/runs/monosdf_room/train_tight.log
LOCK=$ROOT/tmp/gpu.lock
mkdir -p "$ROOT/vid2scene/runs/monosdf_room"

# install the tight conf (committed into the repo, pushed to remote)
cp "$ROOT/vid2scene/scripts/remote/room_grids_tight.conf" "$SRC/code/confs/room_grids_tight.conf"
ls -lh "$SRC/code/confs/room_grids_tight.conf"

# torch.distributed.launch is deprecated in torch 2.2; the agent's working
# pattern was env-var DDP for a single process. Replicate it.
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0
export PIP_CACHE_DIR=$ROOT/pip-cache TORCH_HOME=$ROOT/torch-home HF_HOME=$ROOT/hf-home TMPDIR=$ROOT/tmp
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
# OOM mitigation: the tight scene-bounding-sphere concentrates more SDF
# samples on real surface than the wide-bound run did, and peak memory hit
# the 15.6 GiB ceiling around epoch ~200 in the first attempt. expandable
# segments reduce fragmentation and got us past it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# tcnn/hashencoder build flags + include paths the agent set
SP=$ENV/lib/python3.10/site-packages
export CPATH="$SP/nvidia/cuda_runtime/include:$SP/nvidia/cuda_cccl/include:$SP/nvidia/cusparse/include:$SP/nvidia/cublas/include:$SP/nvidia/curand/include:$CPATH"

cd "$SRC/code"
# screen wrapper already captures stdout/stderr; flock -c just runs the trainer.
# exp_runner.py REQUIRES --local_rank as a CLI arg (legacy torch.distributed.launch
# convention); env-var DDP alone is insufficient. Pass it explicitly.
# --is_continue + --timestamp picks up the latest.pth checkpoint of the named run
# (the previous attempt's run dir). If TIMESTAMP="" we start fresh.
TIMESTAMP="${TIMESTAMP:-}"
EXTRA=""
if [ -n "$TIMESTAMP" ]; then EXTRA="--is_continue --timestamp $TIMESTAMP"; fi
flock "$LOCK" -c "$ENV/bin/python training/exp_runner.py --conf confs/room_grids_tight.conf --local_rank 0 $EXTRA"
echo "MONOSDF_TIGHT_DONE rc=$?"
