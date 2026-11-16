"""Train the encoder from precomputed fingerprint features (result/fps/*.npy)
and report the final verdict + paper metrics. Labels: 0 victim, 1 piracy
(homologous), 2 independent."""
import os

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from torch import optim

from common import DEVICE, INPUT_CHANNEL, RESULT_DIR, set_seed

from encoder import classifier
from encoder_training import train, evaluate, similarity

set_seed(0)
FPS = os.path.join(RESULT_DIR, 'fps')
N_PIRACY = int(os.environ.get('UAP_N_PIRACY', 10))
N_INDEP = int(os.environ.get('UAP_N_INDEP', 10))
ENC_EPOCHS = int(os.environ.get('UAP_ENC_EPOCHS', 25))
ENC_LR = float(os.environ.get('UAP_ENC_LR', 2e-4))


def load(name):
    return torch.from_numpy(np.load(os.path.join(FPS, name + '.npy'))).float()


def make_set(piracy_names, indep_names):
    xs = [load('victim')]
    ys = [torch.zeros(len(xs[0]))]
    for n in piracy_names:
        x = load(n); xs.append(x); ys.append(torch.ones(len(x)))
    for n in indep_names:
        x = load(n); xs.append(x); ys.append(torch.full((len(x),), 2.0))
    return torch.utils.data.TensorDataset(torch.cat(xs), torch.cat(ys).long())


piracy_all = [f'piracy_{i}' for i in range(N_PIRACY)]
indep_all = [f'indep_{i}' for i in range(N_INDEP)]
train_set = make_set(piracy_all[:N_PIRACY // 2], indep_all[:N_INDEP // 2])
fp_v = train_set[0]
print("The length of encoder's training data:", len(train_set))

n_train = int(0.8 * len(train_set))
tr, va = torch.utils.data.random_split(train_set, [n_train, len(train_set) - n_train])
train_loader = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(va, batch_size=64, shuffle=False)

c = classifier(input_channel=INPUT_CHANNEL).to(DEVICE)
train(c, train_loader, val_loader, ENC_EPOCHS, optim.Adam(c.parameters(), lr=ENC_LR), fp_v, 0.2, DEVICE)

print('==========================================')
print('Test (held-out models)')
print('==========================================')
held_piracy, held_indep = piracy_all[N_PIRACY // 2:], indep_all[N_INDEP // 2:]
test_set = make_set(held_piracy, held_indep)
evaluate(c, fp_v, torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False), DEVICE)

# ---- paper metrics ----
c.eval()
with torch.no_grad():
    fp_v_feat = c.feature(fp_v[0].to(DEVICE))
    sims = {}
    for n in held_piracy + held_indep:
        feats = c.feature(load(n).to(DEVICE))
        sims[n] = np.array([similarity(fp_v_feat, feats[i]).item() for i in range(len(feats))])
        print(f'{n}: mean sim {sims[n].mean():.4f}')

pir = np.concatenate([sims[n] for n in held_piracy])
ind = np.concatenate([sims[n] for n in held_indep])
y = np.r_[np.ones(len(pir)), np.zeros(len(ind))]
print(f'\nROC AUC (view level, {len(pir)} vs {len(ind)}): {roc_auc_score(y, np.r_[pir, ind]):.4f}')
means = [sims[n].mean() for n in held_piracy] + [sims[n].mean() for n in held_indep]
labels = [1] * len(held_piracy) + [0] * len(held_indep)
print(f'ROC AUC (model level, {len(held_piracy)} vs {len(held_indep)}): {roc_auc_score(labels, means):.4f}')

rng = np.random.default_rng(0)
print()
for n in held_piracy + held_indep:
    sample = rng.choice(sims[n], 20, replace=False)
    is_p = n in held_piracy
    t, p = stats.ttest_ind(sample, ind if is_p else pir, equal_var=False,
                           alternative='greater' if is_p else 'less')
    print(f'{n}: detection confidence with 20 fingerprints = {(1 - p) * 100:.6f}%  (p = {p:.2e})')

torch.save(c.state_dict(), os.path.join(RESULT_DIR, 'model.pth'))
