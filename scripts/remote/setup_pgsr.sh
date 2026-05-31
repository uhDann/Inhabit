#!/bin/bash
# PGSR (Planar-based Gaussian Splatting Reconstruction) setup.
# zju3dv/PGSR -- forks the 3DGS rasterizer with planar regularization; designed
# to produce smooth TSDF meshes of indoor flat surfaces (walls, table tops, TV
# screens) where vanilla 3D-gaussian-splat depth is noisiest. Apache-2.0 / ZJU
# research license (see the LICENSE file in the cloned repo).
#
# We clone the `monosdf` env because it already has the conda CUDA 12.1 dev
# toolchain (nvcc + headers) wired up; PGSR's submodules need nvcc to build.
# Setup runs CPU-only -- safe to launch in parallel with the MonoSDF training
# that is currently holding the GPU lock; PGSR's *training* will queue behind
# it via flock when we hit `launch_pgsr.sh` later.
set -x
ROOT=/cs/student/projects3/2023/dkozlov
export CONDA_PKGS_DIRS=$ROOT/conda-pkgs
export PIP_CACHE_DIR=$ROOT/pip-cache TMPDIR=$ROOT/tmp
export TORCH_HOME=$ROOT/torch-home HF_HOME=$ROOT/hf-home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME"

CONDA=/cs/student/ug/2023/dkozlov/miniconda3/bin/conda
SRC=$ROOT/conda-envs/monosdf
ENV=$ROOT/conda-envs/pgsr
PY=$ENV/bin/python
PIP=$ENV/bin/pip
REPO=$ROOT/PGSR-src

if [ ! -d "$ENV" ]; then
  echo "=== cloning monosdf -> pgsr (inherits cuda 12.1 nvcc + dev headers) ==="
  "$CONDA" create -y --clone "$SRC" -p "$ENV" || { echo CLONE_FAIL; exit 1; }
fi

if [ ! -d "$REPO" ]; then
  echo "=== cloning PGSR repo ==="
  git clone --recursive https://github.com/zju3dv/PGSR.git "$REPO" || { echo CLONE_REPO_FAIL; exit 1; }
fi
cd "$REPO"
echo "PGSR HEAD: $(git rev-parse --short HEAD)"
echo "submodules:"; ls -d submodules/*/ 2>/dev/null

# Build toolchain env (mirrors what we had to set for MonoSDF's hashencoder)
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4070 Ti SUPER = Ada (sm_89)
SP=$ENV/lib/python3.10/site-packages
export CPATH="$SP/nvidia/cuda_runtime/include:$SP/nvidia/cuda_cccl/include:$SP/nvidia/cusparse/include:$SP/nvidia/cublas/include:$SP/nvidia/curand/include:$CPATH"

"$ENV/bin/nvcc" --version | tail -3

# Python deps PGSR needs but the monosdf env may not have. (PGSR's environment.yml
# lists open3d, plyfile, tqdm, etc.; many overlap MonoSDF, just top up.) Also
# need build-time deps (ninja, setuptools, wheel) for the submodule wheels.
$PIP install --no-input plyfile tqdm tensorboard opencv-python lpips trimesh 2>&1 | tail -4
$PIP install --no-input ninja "setuptools>=68" wheel 2>&1 | tail -4

# Compile the two CUDA submodules. PGSR's planar-aware rasterizer is called
# `diff-plane-rasterization` (NOT `-gaussian-` -- that's vanilla 3DGS). Both
# are vendored as plain directories (no .gitmodules), so a fresh PGSR clone
# already contains them.
# --no-build-isolation is required because the submodules' setup.py imports
# torch at build time and pip's isolated build env doesn't see our torch.
for sub in diff-plane-rasterization simple-knn; do
  if [ -d "submodules/$sub" ]; then
    echo "=== building $sub ==="
    $PIP install --no-input --no-build-isolation "submodules/$sub" 2>&1 | tail -20
  else
    echo "MISSING submodule submodules/$sub"
  fi
done

echo "=== verify imports ==="
$PY -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
import diff_plane_rasterization as dpr
print('diff_plane_rasterization OK')
import simple_knn
print('simple_knn OK')
" 2>&1 | tail -8

echo "PGSR_SETUP_DONE"
