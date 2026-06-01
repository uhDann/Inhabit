#!/bin/bash
# DN-Splatter train + TSDF mesh on MuSHRoom coffee_room (iPhone long_capture),
# using the native `mushroom` dataparser. Behind gpu.lock so it serialises after
# PGSR. normals-from depth (no pretrained-model download), sensor depth loss.
set -x
ROOT=/cs/student/projects3/2023/dkozlov
ENV=$ROOT/conda-envs/dnsplat
DATA=$ROOT/datasets/mushroom/room_datasets/coffee_room
OUT=$ROOT/work/out/dn_mushroom_coffee
LOCK=$ROOT/tmp/gpu.lock
MLOCK=$ROOT/tmp/mesh.lock
mkdir -p "$OUT"

export TMPDIR=$ROOT/tmp PIP_CACHE_DIR=$ROOT/pip-cache
export TORCH_HOME=$ROOT/torch-home HF_HOME=$ROOT/hf-home
export CUDA_VISIBLE_DEVICES=0 PATH=$ENV/bin:$PATH
export PYOPENGL_PLATFORM=egl
cd $ROOT/dn-splatter-src

flock "$LOCK" -c "$ENV/bin/ns-train dn-splatter \
  --pipeline.model.use-depth-loss True \
  --pipeline.model.depth-lambda 0.2 \
  --pipeline.model.use-normal-loss True \
  --pipeline.model.normal-supervision depth \
  --pipeline.datamanager.cache-images cpu \
  --max-num-iterations 30000 \
  --vis tensorboard \
  --output-dir $OUT \
  --experiment-name coffee \
  mushroom --data $DATA --mode iphone --eval-mode within \
  --normals-from depth --depth-mode sensor --load-normals True"
echo "DN_TRAIN_RC=$?"

CFG=$(ls -t $OUT/coffee/dn-splatter/*/config.yml | head -1)
echo "USING CONFIG: $CFG"
MOUT=$OUT/mesh
mkdir -p $MOUT
flock "$MLOCK" -c "$ENV/bin/gs-mesh o3dtsdf --load-config $CFG --output-dir $MOUT \
  --voxel-size 0.01 --sdf-truc 0.03 --depth-trunc 8.0"
echo "DN_MESH_RC=$?"
echo "DN_MUSHROOM_DONE"
