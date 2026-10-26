"""Train and test the fingerprint encoder (mirrors main.py's
framework_building / framework_testing, with local paths and CPU support).

Labels follow the concanate_fingerprint docstring:
  0 = victim, 1 = homologous (piracy) models, 2 = independent models.
Expected result: encoded fingerprints of piracy models are more similar to the
victim's fingerprint than those of independently trained models.
"""
import os
import pickle
import time

import numpy as np
import torch
from torch import optim

from common import CKPT_DIR, DEVICE, INPUT_CHANNEL, N_POINTS, N_VIEWS, RESULT_DIR, fresh_resnet18, set_seed

from encoder import classifier
from encoder_training import train, evaluate
from encoder_trainingdata_preparation import concanate_fingerprint

set_seed(0)

with open(os.path.join(RESULT_DIR, f'split_sets_{N_POINTS}points_{N_VIEWS}views.pkl'), 'rb') as f:
    fingerprint_points = pickle.load(f)

victim = fresh_resnet18()
victim.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'victim.pth'), map_location=DEVICE))
victim.eval()

v_opt = torch.from_numpy(np.load(os.path.join(RESULT_DIR, 'v_opt.npy'))).float().to(DEVICE)


def load_model(name):
    net = fresh_resnet18()
    net.load_state_dict(torch.load(os.path.join(CKPT_DIR, name + '.pth'), map_location=DEVICE))
    net.eval()
    return net


N_PIRACY = int(os.environ.get('UAP_N_PIRACY', 4))
N_INDEP = int(os.environ.get('UAP_N_INDEP', 4))
piracy_all = [f'piracy_{i}' for i in range(N_PIRACY)]
indep_all = [f'indep_{i}' for i in range(N_INDEP)]
piracy_train, piracy_test = piracy_all[:N_PIRACY // 2], piracy_all[N_PIRACY // 2:]
indep_train, indep_test = indep_all[:N_INDEP // 2], indep_all[N_INDEP // 2:]


def build_dataset(piracy_names, indep_names):
    import gc
    data_set = concanate_fingerprint(victim, fingerprint_points, v_opt, 0).dataset
    fp_v = data_set[0]
    for name in piracy_names:
        cur = concanate_fingerprint(load_model(name), fingerprint_points, v_opt, 1).dataset
        data_set = torch.utils.data.ConcatDataset([data_set, cur])
        gc.collect()
    for name in indep_names:
        cur = concanate_fingerprint(load_model(name), fingerprint_points, v_opt, 2).dataset
        data_set = torch.utils.data.ConcatDataset([data_set, cur])
        gc.collect()
    return data_set, fp_v


t0 = time.time()
print('building encoder training data...')
train_set, fp_v = build_dataset(piracy_train, indep_train)
print("The length of encoder's training data:", len(train_set))

n_train = int(0.8 * len(train_set))
train_ds, val_ds = torch.utils.data.random_split(train_set, [n_train, len(train_set) - n_train])
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

ENC_EPOCHS = int(os.environ.get('UAP_ENC_EPOCHS', 10))
ENC_LR = float(os.environ.get('UAP_ENC_LR', 5e-4))
c = classifier(input_channel=INPUT_CHANNEL).to(DEVICE)
optimizer = optim.Adam(c.parameters(), lr=ENC_LR)
train(c, train_loader, val_loader, ENC_EPOCHS, optimizer, fp_v, 0.2, DEVICE)

print('==========================================')
print('Test Begin (held-out piracy and independent models)')
print('==========================================')
test_set, _ = build_dataset(piracy_test, indep_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
evaluate(c, fp_v, test_loader, DEVICE)
print(f'step5 total time {(time.time() - t0) / 60:.1f} min')
