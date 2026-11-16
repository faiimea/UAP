# UAP

This repo is implementation of CVPR 2022 paper ["Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations"](https://ieeexplore.ieee.org/document/9879052)

## About this fork

This fork makes the pipeline runnable end to end on a single GPU. It fixes several bugs in the original code (NaN in the similarity evaluation, an exit-less UAP loop, an overlapping data split, hardcoded CUDA devices and author-machine paths), stabilizes the encoder's contrastive training (best-validation-epoch selection — it collapses late at full scale otherwise), adds a CIFAR-style **ResNet20** to the model zoo, and adds a runnable driver pipeline under `demo/`. Full details in [SETUP_AND_RESULTS.md](SETUP_AND_RESULTS.md).

### Quick start

```bash
# Python 3.10+, then:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # CUDA build for your GPU
pip install numpy scikit-learn scipy tqdm pillow

bash demo/run_full_local.sh   # full pipeline: ResNet20 victim, paper-scale settings
```

Individual steps (`demo/step1..6`) default to a small ResNet18 demo; scale and architecture are set via `UAP_*` environment variables (see `demo/run_full_local.sh`).

### Reproduction results

Full-scale run (ResNet20 victim @ 88.9% CIFAR-10 accuracy, 100 epochs; 10 fine-tuned piracy models, 84.8–87.5%; 10 independent models trained from scratch, 88.1–89.2%; UAP fooling rate 79%; 150 points × 200 views fingerprints; verdicts on 5 piracy + 5 independent models held out of encoder training). Single RTX 5070 Ti Laptop GPU, ~1.5 h total.

| Metric | This fork (ResNet20) | Paper (CIFAR-10) |
|---|---|---|
| ROC AUC — model level | **1.00** (5 vs 5 held-out) | 1.00 |
| ROC AUC — single fingerprint view | **0.97** (1000 vs 1000 views) | ~1.0 |
| Detection confidence with 20 fingerprints | **≥ 99.9999%** for all 10 held-out models (p ≤ 5×10⁻⁷) | > 99.99% |
| Encoded fingerprint similarity to victim | piracy 0.52–0.70 vs independent 0.06–0.41 | separated distributions |

All 10 held-out verdicts correct: every piracy (stolen) model scores above every independently trained model, so a single threshold separates them with no false accusations — with the independents matching the victim's accuracy, so detection rests on decision-boundary shape alone.

## Prerequisites

We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt`

## Directory Structure
```
.
├── encoder.py
├── encoder_trainingdata_preparation.py
├── encoder_training.py
├── fingerprint_point_selection.py
├── main.py
├── model_ext
│   └── test_model_extrac_adv_softlabel.py
├── preparation
│   ├── embedding.py
│   ├── model_ext_2.py
│   ├── model_extrac_adv_softlabel.py
│   ├── model_extraction_cifar10.py
│   ├── normal_adversarial_generation.py
│   ├── simple_extraction_cifar10.py
│   ├── test_model_extrac_adv_softlabel.py
│   └── uap.py
├── README.md
├── test.py
├── train
│   ├── model_structure.py
│   ├── split_subtrain.py
│   ├── train_cifar10_multilabel.py
│   ├── train_cifar10.py
│   └── train_some_models.py
└── utils.py
```

## Run
### Model Preparation
* train_cifar10.py: Train Victim model
* model_extrac_adv_softlabel: Train piracy models
* train_some_models.py: Train homo models

### Fingerprint Generation
* embedding: Gain dataset for fingerprint
* uap: Generate universal adversarial pertubation

### Encoder Training
* main: Train and test of framework


## Citation
Please cite this work if you find it useful:
```bibtex
@inproceedings{peng2022fingerprinting,
  title={Fingerprinting deep neural networks globally via universal adversarial perturbations},
  author={Peng, Zirui and Li, Shaofeng and Chen, Guoxing and Zhang, Cheng and Zhu, Haojin and Xue, Minhui},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={13430--13439},
  year={2022}
}
```
## Paper and GitHub intake

See [paper_intake/README.md](paper_intake/README.md) to extract a paper PDF,
inspect a GitHub repository, and optionally analyze paper evidence with OpenAI.
This standalone tool does not install model dependencies or execute experiments.
