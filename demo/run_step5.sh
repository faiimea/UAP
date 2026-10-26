#!/bin/bash
cd /mnt/c/Users/adars/UAP
export UAP_ARCH=resnet20 UAP_POINTS=150 UAP_VIEWS=200 UAP_N_PIRACY=10 UAP_N_INDEP=10 UAP_ENC_EPOCHS=25 UAP_ENC_LR=2e-4
export CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniforge3/envs/uap/bin/python
for name in victim piracy_0 piracy_1 piracy_2 piracy_3 piracy_4 piracy_5 piracy_6 piracy_7 piracy_8 piracy_9 indep_0 indep_1 indep_2 indep_3 indep_4 indep_5 indep_6 indep_7 indep_8 indep_9; do
  $PY demo/step5a_features.py $name || { echo FAILED > demo/step5_final.status; exit 1; }
done
$PY demo/step5b_encoder.py && echo DONE > demo/step5_final.status || echo FAILED > demo/step5_final.status
