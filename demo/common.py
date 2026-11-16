"""Shared helpers for the reduced-scale demo run of the UAP fingerprinting pipeline.

Everything runs from the repo root (cwd = UAP/). CPU-friendly.
"""
import os
import sys
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'train'))
sys.path.insert(0, os.path.join(REPO, 'preparation'))

from utils import SubTrainDataset  # noqa: E402
from model_structure import get_model  # noqa: E402

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
GPU = torch.cuda.is_available()

# subset-1 statistics used throughout the repo
MEAN = 0.49116027
STD = 0.24728487

DATA_DIR = os.path.join(REPO, 'data')
CKPT_DIR = os.path.join(REPO, 'demo', 'checkpoint')
RESULT_DIR = os.path.join(REPO, 'result')
SPLIT_PKL = os.path.join(DATA_DIR, 'CIFAR10_sub_train_split.pkl')

# scale knobs — overridable via environment for full-scale runs
ARCH = os.environ.get('UAP_ARCH', 'resnet18')
N_POINTS = int(os.environ.get('UAP_POINTS', 30))   # fingerprint generation points (paper: 150)
N_VIEWS = int(os.environ.get('UAP_VIEWS', 50))     # views per point (paper: 200)
INPUT_CHANNEL = 2 * 10 * N_POINTS  # encoder input size

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subset_dataset(idx, train=True):
    """idx: 0 -> victim half, 1 -> attacker half."""
    sets = pickle.load(open(SPLIT_PKL, 'rb'))
    X, y = sets[idx]
    t = train_transform if train else eval_transform
    return SubTrainDataset(X, list(y), transform=t)


def test_loader(batch_size=256, limit=None):
    ds = torchvision.datasets.CIFAR10(root=os.path.join(DATA_DIR, 'CIFAR10'),
                                      train=False, download=True, transform=eval_transform)
    if limit:
        ds = Subset(ds, range(limit))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


@torch.no_grad()
def accuracy(net, loader):
    net.eval().to(DEVICE)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (net(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train_model(net, loader, epochs, lr=0.05, milestones=(4,), log_prefix=''):
    net.to(DEVICE).train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(milestones), gamma=0.2)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        net.train()
        running = 0.0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        print(f'{log_prefix}epoch {ep}/{epochs} avg loss {running / max(1, len(loader)):.4f}', flush=True)
    return net


def fresh_model():
    return get_model(ARCH, GPU, 10)


# backward-compatible alias (older step scripts import this name)
fresh_resnet18 = fresh_model
