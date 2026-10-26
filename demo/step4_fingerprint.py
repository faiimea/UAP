"""Fingerprint point selection (clean version of preparation/embedding.py):
embed the victim half through the victim model, k-means the output-layer
embeddings into N_POINTS clusters, and keep the N_VIEWS nearest neighbours of
one random point per cluster."""
import os
import pickle
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from common import (CKPT_DIR, DEVICE, N_POINTS, N_VIEWS, RESULT_DIR,
                    fresh_resnet18, set_seed, subset_dataset)

set_seed(0)

victim = fresh_resnet18()
victim.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'victim.pth'), map_location=DEVICE))
victim.eval().to(DEVICE)

ds = subset_dataset(0, train=False)  # deterministic transform
loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)

t0 = time.time()
embs, labels, imgs = [], [], []
with torch.no_grad():
    for x, y in loader:
        embs.append(victim(x.to(DEVICE)).cpu())
        labels.append(y)
        imgs.append(x)
embs = torch.cat(embs)          # (25000, 10)
labels = torch.cat(labels)      # (25000,)
imgs = torch.cat(imgs)          # (25000, 3, 32, 32)
print(f'embedded {len(embs)} images in {time.time() - t0:.0f}s')
print('victim self-consistency on its train half:',
      (embs.argmax(1) == labels).float().mean().item())

kmodel = KMeans(n_clusters=N_POINTS, n_init=10, random_state=0)
kmodel.fit(embs.numpy())
ls = kmodel.labels_

nbrs = NearestNeighbors(n_neighbors=N_VIEWS, algorithm='ball_tree').fit(embs.numpy())
_, indices = nbrs.kneighbors(embs.numpy())

data_sets = []
for c in range(N_POINTS):
    members = np.arange(len(ls))[ls == c]
    pick = members[np.random.choice(len(members), 1)[0]]
    knn = indices[pick, :]
    data_img = imgs[knn].numpy().astype(np.float64)          # (N_VIEWS, 3, 32, 32)
    data_label = labels[knn].numpy().reshape(-1, 1).astype(np.float64)
    data_sets.append((data_img, data_label))

out = os.path.join(RESULT_DIR, f'split_sets_{N_POINTS}points_{N_VIEWS}views.pkl')
pickle.dump(data_sets, open(out, 'wb'))
print('saved', out, '| clusters:', len(data_sets), '| views per cluster:', data_sets[0][0].shape[0])
