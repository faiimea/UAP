#!/bin/bash
# One-time environment setup on Gilbreth (run on a login node).
set -e
module purge
module load anaconda 2>/dev/null || module load conda

ENVDIR=$HOME/uap-env
if [ ! -d "$ENVDIR" ]; then
    conda create -y -p "$ENVDIR" python=3.10 pip
fi
"$ENVDIR/bin/pip" install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
"$ENVDIR/bin/pip" install numpy scikit-learn scipy tqdm pillow
"$ENVDIR/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda build:', torch.version.cuda)"
echo "env ready at $ENVDIR"
