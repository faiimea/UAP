#!/bin/bash
# Full-scale run: ResNet20 victim, paper-scale settings. Set PY to override the interpreter
# (e.g. PY=python inside a container).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/uap/bin/python}

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
export UAP_ENC_EPOCHS=25
export UAP_ENC_LR=2e-4

echo "=== step1: data split ==="
[ -f data/CIFAR10_sub_train_split.pkl ] || $PY demo/step1_split.py
echo "=== step2: train model zoo (1 victim + 10 piracy + 10 indep, resnet20) ==="
$PY demo/step2_train_models.py
echo "=== step3: UAP generation ==="
$PY demo/step3_uap.py
echo "=== step4: fingerprint point selection (150x200) ==="
$PY demo/step4_fingerprint.py
echo "=== step5: fingerprint features per model ==="
for name in victim $(seq -f "piracy_%g" 0 $((UAP_N_PIRACY - 1))) $(seq -f "indep_%g" 0 $((UAP_N_INDEP - 1))); do
  $PY demo/step5a_features.py "$name"
done
echo "=== step5: encoder train/test + metrics ==="
$PY demo/step5b_encoder.py
echo "=== FULL RUN DONE ==="
