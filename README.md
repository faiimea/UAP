# UAP

This repo is implementation of CVPR 2022 paper ["Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations"](https://ieeexplore.ieee.org/document/9879052)

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