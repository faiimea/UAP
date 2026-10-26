#!/bin/bash
#SBATCH --job-name=uap-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=uap_full_%j.out
# NOTE: submit with:  sbatch -A <account> [-p <partition>] demo/gilbreth_run.sbatch

set -e
cd "$SLURM_SUBMIT_DIR"
PY=$HOME/uap-env/bin/python

# ---- full-scale configuration (paper settings, ResNet20 victim) ----
export UAP_ARCH=resnet20
export UAP_VICTIM_EPOCHS=100
export UAP_INDEP_EPOCHS=100
export UAP_PIRACY_EPOCHS=10
export UAP_N_PIRACY=10
export UAP_N_INDEP=10
export UAP_POINTS=150
export UAP_VIEWS=200
export UAP_FIT_IMAGES=2000
export UAP_DELTA=0.17      # target 83% fooling rate
export UAP_ROUNDS=10

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== step1: data split ==="
$PY demo/step1_split.py
echo "=== step2: train model zoo (1 victim + 10 piracy + 10 indep) ==="
$PY demo/step2_train_models.py
echo "=== step3: UAP generation ==="
$PY demo/step3_uap.py
echo "=== step4: fingerprint point selection ==="
$PY demo/step4_fingerprint.py
echo "=== step5: encoder train/test ==="
$PY demo/step5_encoder.py
echo "=== step6: metrics ==="
$PY demo/step6_metrics.py
echo "=== DONE ==="
