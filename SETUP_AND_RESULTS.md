# UAP Fingerprinting — Setup & Results

Step-by-step record of getting [faiimea/UAP](https://github.com/faiimea/UAP) (CVPR 2022, *"Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations"*) running on this machine, and the results.

**Machine:** Windows + WSL2 (Ubuntu), RTX 5070 Ti Laptop GPU (12GB, Blackwell/sm_120), 24 CPU cores, 16GB RAM.

---

## 1. Environment setup

The system Python (3.12) had no `pip`/`venv` and is PEP-668 locked, so an isolated Miniforge env was used instead (no root needed):

```bash
# Install Miniforge into the home directory
curl -sSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o mf.sh
bash mf.sh -b -p ~/miniforge3 && rm mf.sh

# Python 3.10 env
~/miniforge3/bin/mamba create -n uap python=3.10 pip -y
```

**GPU gotcha:** the RTX 5070 Ti is Blackwell (sm_120) and requires PyTorch ≥ 2.7 built for CUDA 12.8 — the `torch==2.2.0` pinned in `requirements.txt` cannot use this GPU. Also, plain `nvidia-smi` can transiently fail in WSL2 with *"Failed to initialize NVML"*; the reliable binary is `/usr/lib/wsl/lib/nvidia-smi`.

```bash
~/miniforge3/envs/uap/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0
~/miniforge3/envs/uap/bin/pip install numpy==1.24.3 scikit-learn==1.3.2 tqdm matplotlib pillow tensorboard
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## 2. Code fixes required

The repo is research code, hardcoded to the authors' server. Changes made:

| File | Change | Why |
|---|---|---|
| `preparation/uap.py` | `.cuda()` / `torch.cuda.set_device(1)` → device-aware `.to(DEVICE)`; added `max_rounds` parameter; guarded a division by zero | Ran only on the authors' multi-GPU box; the UAP loop had no iteration cap and could run forever |
| `encoder_trainingdata_preparation.py` | `.cuda()` → `.to(DEVICE)` | Same |
| `encoder_training.py` | Save path `/home/fazhong/...` → `./result/model.pth`; **added `+1e-8` to both cosine-similarity denominators** | The epsilon fixes a real bug: held-out fingerprints can produce all-zero ReLU features → 0/0 → the final evaluation printed `NaN` |
| `demo/` (new) | 5 runnable driver scripts replacing `main.py` | `main.py` has a broken import (`uap.train.model_structure`) and absolute paths to checkpoints that don't ship with the repo |

⚠️ **Naming trap:** in the authors' `main.py`, the `indep_*` variables actually load extracted (piracy) models and `homo_*` loads independent ones — swapped relative to the paper. `demo/step5_encoder.py` follows the documented label semantics (0 = victim, 1 = homologous/piracy, 2 = independent), so its printed lines mean what they say.

## 3. Running the pipeline

No pre-trained checkpoints ship with the repo, so everything is trained from scratch. All commands from the repo root:

```bash
PY=~/miniforge3/envs/uap/bin/python
$PY demo/step1_split.py        # download CIFAR-10, split into disjoint victim/attacker halves
$PY demo/step2_train_models.py # train the 9-model zoo            (~11 min on GPU)
$PY demo/step3_uap.py          # generate the UAP (DeepFool)      (~1 min)
$PY demo/step4_fingerprint.py  # select fingerprint points        (~1 min)
$PY demo/step5_encoder.py      # train + test the encoder         (~1.5 min)
```

(Step 1 also fixes a bug in the original `train/split_subtrain.py`, which used `~vic_idx` — bitwise NOT on integer indices, not a set complement — producing overlapping halves.)

Scale used (vs. paper): victim 20 epochs (paper ~100), 4+4 suspect models, fingerprints 30 points × 50 views (paper 150 × 200). Knobs live in `demo/common.py` and `demo/step2_train_models.py`.

## 4. Results

### Model zoo (CIFAR-10 test accuracy)

| Model | Role | Trained on | Accuracy |
|---|---|---|---|
| `victim` | victim ResNet18 | victim half, 20 epochs | **87.5%** |
| `piracy_0..3` | fine-tuned copies of victim (homologous) | attacker half, 3 epochs from victim weights | 85.6 – 86.8% |
| `indep_0..3` | independent ResNet18s | attacker half, 12 epochs from scratch | 83.5 – 84.4% |

### Universal adversarial perturbation

Fitted on 500 victim-half images, validated on 2,000 test images: fooling rate **27.6% → 40.7% → 46.9%** over 3 rounds (L∞ = 0.63 in normalized space). Saved to `result/v_opt.npy`.

### Fingerprint verification (the headline result)

The encoder was trained on the victim + 2 piracy + 2 independent models, then tested on the **held-out** 2 piracy + 2 independent models it never saw:

| Suspect type | Fingerprint similarity to victim |
|---|---|
| Piracy (derived from victim) | **0.419** |
| Independent (innocent) | **0.017** |

A ~25× separation: models derived from the victim are clearly flagged, while independently trained models on the same task/architecture are not — reproducing the paper's core claim at reduced scale. (The paper reports near-perfect ARUC at full scale — 100-epoch models, 150×200 fingerprints, larger model zoos including extraction attacks and other architectures.)

## 5. Artifacts on disk

- `demo/checkpoint/*.pth` — the 9 trained models
- `result/v_opt.npy` — the universal adversarial perturbation
- `result/split_sets_30points_50views.pkl` — fingerprint points
- `result/model.pth` — the trained fingerprint encoder
- `demo/step2.log`, `demo/step3.log` — training/generation logs
