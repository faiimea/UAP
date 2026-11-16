"""Generate the universal adversarial perturbation against the victim
using the repo's DeepFool-based universal_perturbation (preparation/uap.py),
at reduced scale for CPU."""
import os
import time

import numpy as np
import torch
from torch.utils.data import Subset

from common import CKPT_DIR, DEVICE, RESULT_DIR, fresh_resnet18, set_seed, subset_dataset, test_loader

from uap import universal_perturbation  # repo's preparation/uap.py (patched for CPU)

set_seed(0)

victim = fresh_resnet18()
victim.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'victim.pth'), map_location=DEVICE))
victim.eval()

# pool of images to fit the perturbation on, deterministic transform
FIT_IMAGES = int(os.environ.get('UAP_FIT_IMAGES', 500))
DELTA = float(os.environ.get('UAP_DELTA', 0.4))
ROUNDS = int(os.environ.get('UAP_ROUNDS', 3))
train_ds = Subset(subset_dataset(0, train=False), range(FIT_IMAGES))
val_loader = test_loader(batch_size=128, limit=2000)

t0 = time.time()
v_opt, foo_rates, att_rates = universal_perturbation(
    train_ds, val_loader, victim,
    delta=DELTA,        # stop once fooling rate on val exceeds 1-delta
    xi=10,
    num_classes=10,
    max_iter_df=20,
    max_rounds=ROUNDS,
)
print(f'UAP generation took {(time.time() - t0) / 60:.1f} min')
print('fooling rates per round:', foo_rates)
print('attack rates per round:', att_rates)

out = os.path.join(RESULT_DIR, 'v_opt.npy')
np.save(out, v_opt)
print('saved', out, '| shape', np.asarray(v_opt).shape, '| Linf', np.abs(v_opt).max())
