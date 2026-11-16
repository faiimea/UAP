# UAP

This repo is implementation of CVPR 2022 paper ["Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations"](https://ieeexplore.ieee.org/document/9879052)

## About this fork

This fork contains two explicitly separate paths. The legacy `demo/step*.py`
path is a reduced single-architecture smoke/research demo. The
`demo/paper_full_cifar10.py` path is the paper-faithful CIFAR-10 baseline used
by the LangChain agent: one ResNet18 victim, 10 adversarially extracted piracy
models and 10 homologous models for each of ResNet34, VGG16, DenseNet121 and
GoogLeNet (81 models in total), the UAP, 100-point/200-view fingerprints,
encoder training, metrics, and validation. Full details in
[SETUP_AND_RESULTS.md](SETUP_AND_RESULTS.md).

### Quick start

```bash
# Python 3.10+, then:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # CUDA build for your GPU
pip install numpy scikit-learn scipy tqdm pillow

bash demo/run_full_local.sh   # legacy reduced pipeline

# paper-faithful pipeline through the reproduction tools
python -m paper_tools.langchain_agent \
  --spec papers/uap/paper_faithful_from_scratch_spec.yaml \
  --workspace inspection-results/paper-faithful-run \
  --prompt "Run the complete paper-faithful UAP reproduction, then reproduce and validate every metric."
```

Individual steps (`demo/step1..6`) default to a small ResNet18 demo; scale and architecture are set via `UAP_*` environment variables (see `demo/run_full_local.sh`).

### Legacy reduced-demo results

The numbers below are from the legacy reduced demo, not from the
paper-faithful 81-model path. They are useful as a quick sanity check only.
The paper-faithful run writes its own metrics and manifest under
`result/paper_full_cifar10/` and must be run before claiming a paper-level
reproduction.

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
