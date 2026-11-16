#!/bin/bash
# Train the checkpoint layout required by papers/uap/spec.yaml, then run the
# full demo pipeline. This is the preparation job for the actual Docker
# reproduce_metric test; it is intentionally separate from the reuse-model job.
#
# Submit from the repository root:
#   sbatch -A zghodsi -p a100-80gb --qos=normal demo/gilbreth_train_for_docker.sbatch
#SBATCH --job-name=uap-docker-inputs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=uap_docker_inputs_%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
PY="${PY:-$HOME/uap-env/bin/python}"

export UAP_ARCH=resnet20
export UAP_VICTIM_EPOCHS=100
export UAP_INDEP_EPOCHS=100
export UAP_PIRACY_EPOCHS=10
export UAP_N_PIRACY=10
export UAP_N_INDEP=10
export UAP_POINTS=150
export UAP_VIEWS=200
export UAP_FIT_IMAGES=2000
export UAP_DELTA=0.17
export UAP_ROUNDS=10
export UAP_ENC_EPOCHS=25
export UAP_ENC_LR=2e-4

echo "=== job ${SLURM_JOB_ID:-unknown} on $(hostname) ==="
nvidia-smi
"$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

if [[ ! -f data/CIFAR10_sub_train_split.pkl ]]; then
  "$PY" demo/step1_split.py
fi

echo "=== step2: train exact Docker input model zoo ==="
"$PY" demo/step2_train_models.py

for name in victim $(seq -f "piracy_%g" 0 9) $(seq -f "indep_%g" 0 9); do
  [[ -f "demo/checkpoint/${name}.pth" ]] || { echo "Missing ${name}.pth" >&2; exit 2; }
done

echo "=== step3: UAP ==="
"$PY" demo/step3_uap.py
echo "=== step4: fingerprint points ==="
"$PY" demo/step4_fingerprint.py
echo "=== step5a: features ==="
for name in victim $(seq -f "piracy_%g" 0 9) $(seq -f "indep_%g" 0 9); do
  "$PY" demo/step5a_features.py "$name"
done
echo "=== step5b: encoder ==="
"$PY" demo/step5b_encoder.py
echo "=== step6: metrics ==="
"$PY" demo/step6_metrics.py | tee "demo/metrics_${SLURM_JOB_ID:-local}.txt"

echo "=== Docker input checkpoint files ==="
find demo/checkpoint -maxdepth 1 -type f -name '*.pth' -printf '%f\n' | sort
echo "=== COMPLETE: copy demo/checkpoint to the Docker host ==="
