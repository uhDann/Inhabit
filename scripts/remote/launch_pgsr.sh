#!/bin/bash
# PGSR train + TSDF mesh on the Mip-NeRF 360 `room` data. Behind the gpu.lock
# so we don't OOM the MonoSDF training currently using the GPU; flock will
# block until MonoSDF tight releases.
set -x
ROOT=/cs/student/projects3/2023/dkozlov
ENV=$ROOT/conda-envs/pgsr
REPO=$ROOT/PGSR-src
DATA=$ROOT/datasets/mip360/room
OUT=$ROOT/vid2scene/runs/pgsr_room
LOCK=$ROOT/tmp/gpu.lock
mkdir -p "$OUT"

export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
export TMPDIR=$ROOT/tmp PIP_CACHE_DIR=$ROOT/pip-cache
export TORCH_HOME=$ROOT/torch-home HF_HOME=$ROOT/hf-home
export CUDA_VISIBLE_DEVICES=0
SP=$ENV/lib/python3.10/site-packages
export CPATH="$SP/nvidia/cuda_runtime/include:$SP/nvidia/cuda_cccl/include:$SP/nvidia/cusparse/include:$SP/nvidia/cublas/include:$SP/nvidia/curand/include:$CPATH"

cd "$REPO"

# Train (PGSR README's indoor-scene flags: max_abs_split_points 0 +
# opacity_cull_threshold 0.05 for clean planar splatting).
flock "$LOCK" -c "$ENV/bin/python train.py -s $DATA -m $OUT --max_abs_split_points 0 --opacity_cull_threshold 0.05 --resolution 4"
echo "PGSR_TRAIN_RC=$?"

# TSDF mesh extraction (uses the trained model; --voxel_size controls mesh
# fineness; --max_depth bounds far surfaces).
flock "$LOCK" -c "$ENV/bin/python render.py -m $OUT --max_depth 5.0 --voxel_size 0.01"
echo "PGSR_MESH_RC=$?"

echo "PGSR_DONE"
