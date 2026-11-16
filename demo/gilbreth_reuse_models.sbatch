#!/bin/bash
# Run the non-training portion of the full UAP experiment on Gilbreth.
# Submit from the repository root with, for example:
#   sbatch -A YOUR_ACCOUNT --partition=a30 demo/gilbreth_reuse_models.sbatch
# Add --partition=a100-40gb or another partition available to your group if needed.
#SBATCH --job-name=uap-reuse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=uap_reuse_%j.out
#
# The script expects the trained model zoo at demo/checkpoint/:
#   victim.pth, piracy_0.pth ... piracy_9.pth,
#   indep_0.pth ... indep_9.pth
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
PY="${PY:-$HOME/uap-env/bin/python}"

export UAP_ARCH=resnet20
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
"$PY" -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

if [[ ! -f data/CIFAR10_sub_train_split.pkl ]]; then
  echo "=== step1: data split ==="
  "$PY" demo/step1_split.py
fi

for name in victim $(seq -f "piracy_%g" 0 9) $(seq -f "indep_%g" 0 9); do
  [[ -f "demo/checkpoint/${name}.pth" ]] || { echo "Missing checkpoint: demo/checkpoint/${name}.pth" >&2; exit 2; }
done

if [[ ! -f result/v_opt.npy ]]; then
  echo "=== step3: UAP generation ==="
  "$PY" demo/step3_uap.py
else
  echo "=== step3: result/v_opt.npy exists; skipping ==="
fi

if [[ ! -f result/split_sets_${UAP_POINTS}points_${UAP_VIEWS}views.pkl ]]; then
  echo "=== step4: fingerprint point selection ==="
  "$PY" demo/step4_fingerprint.py
else
  echo "=== step4: fingerprint split exists; skipping ==="
fi

echo "=== step5a: fingerprint features ==="
for name in victim $(seq -f "piracy_%g" 0 9) $(seq -f "indep_%g" 0 9); do
  "$PY" demo/step5a_features.py "$name"
done

if [[ ! -f result/model.pth ]]; then
  echo "=== step5b: encoder training and held-out metrics ==="
  "$PY" demo/step5b_encoder.py
else
  echo "=== step5b: result/model.pth exists; skipping encoder training ==="
fi

echo "=== step6: final metrics ==="
"$PY" demo/step6_metrics.py | tee "demo/metrics_${SLURM_JOB_ID:-local}.txt"
echo "=== UAP REUSE-MODEL RUN DONE ==="
