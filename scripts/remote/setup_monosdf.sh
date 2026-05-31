#!/bin/bash
# MonoSDF setup: clone `splat` env, install a MINIMAL CUDA 12.1 dev toolchain
# (nvcc + dev headers only -- avoids the cuda-gdb metapackage that broke conda
# before), then build tiny-cuda-nn for sm_89 (RTX 4070 Ti SUPER = Ada).
# If tcnn build fails we fall back to MonoSDF's pure-PyTorch MLP config.
set -x
ROOT=/cs/student/projects3/2023/dkozlov
export CONDA_PKGS_DIRS=$ROOT/conda-pkgs
export PIP_CACHE_DIR=$ROOT/pip-cache
export TMPDIR=$ROOT/tmp
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
CONDA=/cs/student/ug/2023/dkozlov/miniconda3/bin/conda
SRC=$ROOT/conda-envs/splat
ENV=$ROOT/conda-envs/monosdf
PY=$ENV/bin/python
PIP=$ENV/bin/pip

if [ ! -d "$ENV" ]; then
  echo "=== cloning splat -> monosdf ==="
  "$CONDA" create -y --clone "$SRC" -p "$ENV" || { echo CLONE_FAIL; exit 1; }
fi

echo "=== installing minimal CUDA 12.1 dev toolchain (no gdb metapackage) ==="
"$CONDA" install -y -p "$ENV" -c nvidia/label/cuda-12.1.0 \
  cuda-nvcc cuda-cudart-dev cuda-cccl libcurand-dev cuda-nvrtc-dev cuda-profiler-api 2>&1 | tail -15

export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
echo "=== nvcc check ==="
"$ENV/bin/nvcc" --version | tail -3 && echo NVCC_OK || { echo NVCC_FAIL; }

echo "=== building tiny-cuda-nn (sm_89) ==="
export TCNN_CUDA_ARCHITECTURES=89
$PIP install --no-input "ninja" 2>&1 | tail -2
$PIP install --no-input "git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch" 2>&1 | tail -25
$PY -c "import tinycudann as tcnn; print('TCNN_OK', tcnn.__version__ if hasattr(tcnn,'__version__') else 'imported')" 2>&1 | tail -3

echo "=== monosdf python deps ==="
$PIP install --no-input trimesh pyhocon opencv-python scikit-image PyMCubes plyfile 2>&1 | tail -8
echo MONOSDF_SETUP_DONE
