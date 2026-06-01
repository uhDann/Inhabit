#!/bin/bash
# PGSR train + TSDF mesh on MuSHRoom coffee_room (iPhone long_capture).
# COLMAP sparse/0 prebuilt by build_colmap_mushroom.py. Behind gpu.lock.
set -x
ROOT=/cs/student/projects3/2023/dkozlov
ENV=$ROOT/conda-envs/pgsr
REPO=$ROOT/PGSR-src
DATA=$ROOT/work/mushroom_coffee_colmap
OUT=$ROOT/work/out/pgsr_mushroom_coffee
LOCK=$ROOT/tmp/gpu.lock
mkdir -p "$OUT"

export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
export TMPDIR=$ROOT/tmp PIP_CACHE_DIR=$ROOT/pip-cache
export TORCH_HOME=$ROOT/torch-home HF_HOME=$ROOT/hf-home CUDA_VISIBLE_DEVICES=0
SP=$ENV/lib/python3.10/site-packages
export CPATH="$SP/nvidia/cuda_runtime/include:$SP/nvidia/cuda_cccl/include:$SP/nvidia/cusparse/include:$SP/nvidia/cublas/include:$SP/nvidia/curand/include:$CPATH"
cd "$REPO"

# indoor flags (planar-clean), native res, 30k iters like the paper's indoor runs
flock "$LOCK" -c "$ENV/bin/python train.py -s $DATA -m $OUT \
  --max_abs_split_points 0 --opacity_cull_threshold 0.05 \
  --resolution 1 --iterations 30000 --test_iterations 30000 --save_iterations 30000"
echo "PGSR_TRAIN_RC=$?"

# TSDF mesh: real metric scale, indoor depth bound
flock "$LOCK" -c "$ENV/bin/python render.py -m $OUT --max_depth 8.0 --voxel_size 0.01"
echo "PGSR_MESH_RC=$?"
echo "PGSR_MUSHROOM_DONE"
