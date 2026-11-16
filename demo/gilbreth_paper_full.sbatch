#!/bin/bash
# Full paper-faithful CIFAR-10 UAP reproduction for the Slurm fallback.
# Submit through paper_tools; do not run this on the login node directly.
#SBATCH --job-name=uap-paper-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=uap_paper_full_%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
PY="${PY:-$HOME/uap-env/bin/python}"
export UAP_WORKERS="${UAP_WORKERS:-8}"
export UAP_POINTS="${UAP_POINTS:-100}"
export UAP_VIEWS="${UAP_VIEWS:-200}"
export UAP_FIT_IMAGES="${UAP_FIT_IMAGES:-2000}"
export UAP_VAL_IMAGES="${UAP_VAL_IMAGES:-2000}"
export UAP_DELTA="${UAP_DELTA:-0.17}"
export UAP_MAX_ROUNDS="${UAP_MAX_ROUNDS:-1000}"
export UAP_ENC_BATCH="${UAP_ENC_BATCH:-512}"

echo "=== job ${SLURM_JOB_ID:-unknown} on $(hostname) ==="
nvidia-smi
"$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
mkdir -p result/paper_full_cifar10
"$PY" demo/paper_full_cifar10.py | tee "result/paper_full_cifar10/metrics_${SLURM_JOB_ID:-local}.log"
cp result/paper_full_cifar10/manifest.json "result/paper_full_cifar10/manifest_${SLURM_JOB_ID:-local}.json"
