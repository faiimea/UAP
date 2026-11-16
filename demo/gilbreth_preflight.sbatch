#!/bin/bash
# Short GPU preflight: validates the environment and all 21 trained models.
# Submit from the repository root with:
#   sbatch -A YOUR_ACCOUNT --partition=YOUR_PARTITION demo/gilbreth_preflight.sbatch
#SBATCH --job-name=uap-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=uap_preflight_%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
PY="${PY:-$HOME/uap-env/bin/python}"
export UAP_ARCH=resnet20

echo "=== preflight job ${SLURM_JOB_ID:-unknown} on $(hostname) ==="
nvidia-smi

"$PY" - <<'PY'
import os
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device="cuda")
torch.cuda.synchronize()
print("CUDA operation: OK", float((x @ x).mean()))

from demo.common import fresh_model

names = ["victim"] + [f"piracy_{i}" for i in range(10)] + [f"indep_{i}" for i in range(10)]
for name in names:
    path = os.path.join("demo", "checkpoint", name + ".pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    model = fresh_model()
    model.load_state_dict(torch.load(path, map_location="cuda"))
    del model
    print("checkpoint: OK", name)

print("PREFLIGHT PASSED")
PY
