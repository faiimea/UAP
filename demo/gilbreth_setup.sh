#!/bin/bash
# One-time environment setup on Gilbreth (run on a login node).
set -euo pipefail
module purge

ENVDIR=$HOME/uap-env
loaded_uv=0
if ! command -v uv >/dev/null 2>&1; then
    for mod in uv/0.8.0 uv/0.7.0; do
        if module load "$mod" >/dev/null 2>&1; then
            loaded_uv=1
            break
        fi
    done
fi

if command -v uv >/dev/null 2>&1; then
    echo "Using uv: $(command -v uv)"
    if [ ! -x "$ENVDIR/bin/python" ]; then
        uv venv --python 3.10 "$ENVDIR"
    fi
    uv pip install --python "$ENVDIR/bin/python" \
        --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.2.0" "torchvision==0.17.0"
    uv pip install --python "$ENVDIR/bin/python" \
        "numpy==1.26.4" scikit-learn scipy tqdm pillow
else
    # Fallback for sites where uv is unavailable: try versioned Miniforge
    # modules. `module spider conda` shows the versions exposed by RCAC.
    loaded_conda=0
    for mod in conda/2026.03 conda/2025.09 conda/2025.02 conda/2024.09; do
        if module load "$mod" >/dev/null 2>&1; then
            loaded_conda=1
            break
        fi
    done
    if [ "$loaded_conda" -ne 1 ] || ! command -v conda >/dev/null 2>&1; then
        echo "Could not load uv or a versioned conda module." >&2
        exit 1
    fi
    if [ ! -x "$ENVDIR/bin/python" ]; then
        conda create -y -p "$ENVDIR" python=3.10 pip
    fi
    "$ENVDIR/bin/pip" install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
    "$ENVDIR/bin/pip" install "numpy==1.26.4" scikit-learn scipy tqdm pillow
fi
"$ENVDIR/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda build:', torch.version.cuda)"
echo "env ready at $ENVDIR"
