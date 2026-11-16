"""Compute the paper's evaluation metrics for the held-out suspect models:
- per-view cosine similarities of encoder representations to the victim's
- ROC AUC (piracy = positive, independent = negative)
- hypothesis-test detection confidence with 20 fingerprint views (Welch t-test)
"""
import os
import pickle

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import CKPT_DIR, DEVICE, INPUT_CHANNEL, N_POINTS, N_VIEWS, RESULT_DIR, fresh_resnet18, set_seed

from encoder import classifier
from encoder_training import similarity
from encoder_trainingdata_preparation import concanate_fingerprint

set_seed(0)

with open(os.path.join(RESULT_DIR, f'split_sets_{N_POINTS}points_{N_VIEWS}views.pkl'), 'rb') as f:
    fingerprint_points = pickle.load(f)

victim = fresh_resnet18()
victim.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'victim.pth'), map_location=DEVICE))
victim.eval()

v_opt = torch.from_numpy(np.load(os.path.join(RESULT_DIR, 'v_opt.npy'))).float().to(DEVICE)

enc = classifier(input_channel=INPUT_CHANNEL).to(DEVICE)
enc.load_state_dict(torch.load(os.path.join(RESULT_DIR, 'model.pth'), map_location=DEVICE))
enc.eval()


def load_model(name):
    net = fresh_resnet18()
    net.load_state_dict(torch.load(os.path.join(CKPT_DIR, name + '.pth'), map_location=DEVICE))
    net.eval()
    return net


@torch.no_grad()
def view_similarities(net):
    """Similarity of each of the model's fingerprint views to the victim's."""
    ds = concanate_fingerprint(net, fingerprint_points, v_opt, 0).dataset
    fps = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEVICE)
    feats = enc.feature(fps)
    return np.array([similarity(fp_v_feat, feats[i]).item() for i in range(feats.shape[0])])


with torch.no_grad():
    vic_ds = concanate_fingerprint(victim, fingerprint_points, v_opt, 0).dataset
    fp_v_feat = enc.feature(vic_ds[0][0].to(DEVICE))

N_PIRACY = int(os.environ.get('UAP_N_PIRACY', 4))
N_INDEP = int(os.environ.get('UAP_N_INDEP', 4))
held_out_piracy = [f'piracy_{i}' for i in range(N_PIRACY // 2, N_PIRACY)]
held_out_indep = [f'indep_{i}' for i in range(N_INDEP // 2, N_INDEP)]

sims = {}
for name in held_out_piracy + held_out_indep:
    sims[name] = view_similarities(load_model(name))
    print(f'{name}: mean sim {sims[name].mean():.4f}  (min {sims[name].min():.4f}, max {sims[name].max():.4f})')

pir = np.concatenate([sims[n] for n in held_out_piracy])
ind = np.concatenate([sims[n] for n in held_out_indep])

# ROC AUC over all held-out fingerprint views (paper reports AUC 1.0/1.0/0.98)
y = np.r_[np.ones(len(pir)), np.zeros(len(ind))]
auc = roc_auc_score(y, np.r_[pir, ind])
print(f'\nROC AUC (view level, {len(pir)} piracy vs {len(ind)} independent views): {auc:.4f}')

# per-model AUC (4 models)
model_means = [sims[n].mean() for n in held_out_piracy] + [sims[n].mean() for n in held_out_indep]
model_labels = [1] * len(held_out_piracy) + [0] * len(held_out_indep)
auc_model = roc_auc_score(model_labels, model_means)
print(f'ROC AUC (model level, {len(held_out_piracy)} vs {len(held_out_indep)}): {auc_model:.4f}')

# Detection confidence with 20 fingerprints (paper: >99.99%): for each piracy
# model, Welch t-test of 20 of its view-similarities against the independent
# models' distribution; confidence = 1 - p (one-sided, greater).
print()
rng = np.random.default_rng(0)
for name in held_out_piracy + held_out_indep:
    sample = rng.choice(sims[name], 20, replace=False)
    t, p = stats.ttest_ind(sample, ind if name in held_out_piracy else pir,
                           equal_var=False, alternative='greater' if name in held_out_piracy else 'less')
    print(f'{name}: detection confidence with 20 fingerprints = {(1 - p) * 100:.6f}%  (p = {p:.2e})')

# threshold analysis
lo, hi = ind.max(), pir.min()
print(f'\nseparation gap: all independent views <= {lo:.4f}, all piracy views >= {hi:.4f}'
      if hi > lo else f'\ndistributions overlap: indep max {lo:.4f} > piracy min {hi:.4f}')
