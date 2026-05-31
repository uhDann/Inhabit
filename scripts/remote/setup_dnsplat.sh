#!/bin/bash
# DN-Splatter setup: clone the working `splat` env, add nerfstudio + dn-splatter.
# No compilation needed (prebuilt gsplat wheel already in splat env).
set -x
ROOT=/cs/student/projects3/2023/dkozlov
export CONDA_PKGS_DIRS=$ROOT/conda-pkgs
export PIP_CACHE_DIR=$ROOT/pip-cache
export TMPDIR=$ROOT/tmp
export TORCH_HOME=$ROOT/torch-home
export HF_HOME=$ROOT/hf-home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME"
CONDA=/cs/student/ug/2023/dkozlov/miniconda3/bin/conda
SRC=$ROOT/conda-envs/splat
ENV=$ROOT/conda-envs/dnsplat
PY=$ENV/bin/python
PIP=$ENV/bin/pip

if [ ! -d "$ENV" ]; then
  echo "=== cloning splat -> dnsplat ==="
  "$CONDA" create -y --clone "$SRC" -p "$ENV" || { echo CLONE_FAIL; exit 1; }
fi

# record the prebuilt gsplat wheel version so we can restore if pip clobbers it
GSV=$($PY -c "import gsplat;print(gsplat.__version__)" 2>/dev/null)
echo "gsplat before: $GSV"

# nerfstudio (pulls tyro/viser/etc; uses existing torch 2.2 cu121). dn-splatter on top.
$PIP install --no-input "nerfstudio" 2>&1 | tail -20
$PIP install --no-input "git+https://github.com/maturk/dn-splatter.git" 2>&1 | tail -20

# guard: if pip rebuilt/removed the prebuilt gsplat, reinstall the working wheel
$PY -c "import gsplat" 2>/dev/null || $PIP install --no-input --force-reinstall --no-deps "gsplat==$GSV" 2>&1 | tail -5

echo "=== verify ==="
$PY -c "import nerfstudio,gsplat,torch;print('OK ns',nerfstudio.__version__,'gsplat',gsplat.__version__,'torch',torch.__version__)" 2>&1 | tail -5
$ENV/bin/ns-train --help >/dev/null 2>&1 && echo NS_TRAIN_OK || echo NS_TRAIN_MISSING
echo DNSPLAT_SETUP_DONE
