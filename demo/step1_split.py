"""Split CIFAR-10 train set into two disjoint halves (victim / attacker).

Same as train/split_subtrain.py but with the complement computed correctly
(the original used `~vic_idx` on an integer index array, which is bitwise NOT,
not a set complement).
"""
import os
import pickle

import numpy as np
import torchvision

from common import DATA_DIR, SPLIT_PKL

np.random.seed(0)

trainset = torchvision.datasets.CIFAR10(root=os.path.join(DATA_DIR, 'CIFAR10'),
                                        train=True, download=True, transform=None)
n = len(trainset)
vic_idx = np.random.choice(n, n // 2, replace=False)
att_idx = np.setdiff1d(np.arange(n), vic_idx)

X = trainset.data                      # (50000, 32, 32, 3) uint8
y = np.array(trainset.targets)

pickle.dump(((X[vic_idx], y[vic_idx]), (X[att_idx], y[att_idx])), open(SPLIT_PKL, 'wb'))
print('saved', SPLIT_PKL, '| victim half:', len(vic_idx), '| attacker half:', len(att_idx))
