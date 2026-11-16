"""Compute one model's fingerprint feature matrix (n_views x 2*10*n_points)
and save it to result/fps/<name>.npy. Low-memory version: CPU-only, probe
images memory-mapped from result/fp_points.npy (built on first run from the
step4 pickle). Feature layout matches concanate_fingerprint exactly:
per view, per probe image: [10 clean softmax, 10 perturbed softmax]."""
import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn

from common import CKPT_DIR, DEVICE, N_POINTS, N_VIEWS, RESULT_DIR, fresh_model

name = sys.argv[1]
out_dir = os.path.join(RESULT_DIR, 'fps')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, name + '.npy')
if os.path.exists(out):
    print(f'{out} exists, skipping', flush=True)
    sys.exit(0)

# one-time: pickle -> plain npy so later runs can memory-map it
points_npy = os.path.join(RESULT_DIR, 'fp_points.npy')
if not os.path.exists(points_npy):
    with open(os.path.join(RESULT_DIR, f'split_sets_{N_POINTS}points_{N_VIEWS}views.pkl'), 'rb') as f:
        ds = pickle.load(f)
    arr = np.stack([img.astype(np.float32) for img, _ in ds])  # (points, views, 3, 32, 32)
    np.save(points_npy, arr)
    del ds, arr

points = np.load(points_npy, mmap_mode='r')  # (N_POINTS, N_VIEWS, 3, 32, 32)
v_opt = torch.from_numpy(np.load(os.path.join(RESULT_DIR, 'v_opt.npy'))).float().to(DEVICE)

net = fresh_model()
net.load_state_dict(torch.load(os.path.join(CKPT_DIR, name + '.pth'), map_location=DEVICE))
net.eval()
softmax = nn.Softmax(dim=1)

features = np.zeros((N_VIEWS, 2 * 10 * N_POINTS), dtype=np.float32)
with torch.no_grad():
    for j in range(N_VIEWS):
        x = torch.from_numpy(np.ascontiguousarray(points[:, j])).to(DEVICE)  # (N_POINTS, 3, 32, 32)
        clean = softmax(net(x).cpu())
        pert = softmax(net(x + v_opt).cpu())
        features[j] = torch.cat((clean, pert), 1).reshape(-1).numpy()

np.save(out, features)
print(f'{name}: saved {features.shape} -> {out}', flush=True)
