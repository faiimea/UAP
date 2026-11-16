"""Train the model zoo (reduced scale, CPU):

- victim:   ResNet18 on the victim half of CIFAR-10
- piracy_i: fine-tuned copies of the victim (homologous / stolen models)
- indep_i:  ResNet18 trained from scratch on the attacker half (independent models)
"""
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from common import (CKPT_DIR, DEVICE, fresh_resnet18, set_seed, subset_dataset,
                    test_loader, train_model, accuracy)

VICTIM_EPOCHS = int(os.environ.get('UAP_VICTIM_EPOCHS', 20))
INDEP_EPOCHS = int(os.environ.get('UAP_INDEP_EPOCHS', 12))
PIRACY_EPOCHS = int(os.environ.get('UAP_PIRACY_EPOCHS', 3))
N_PIRACY = int(os.environ.get('UAP_N_PIRACY', 4))
N_INDEP = int(os.environ.get('UAP_N_INDEP', 4))


def milestones_for(epochs):
    return (max(1, int(epochs * 0.6)), max(2, int(epochs * 0.85)))

tl = test_loader()


def save(net, name):
    path = os.path.join(CKPT_DIR, name + '.pth')
    torch.save(net.state_dict(), path)
    acc = accuracy(net, tl)
    print(f'[{name}] test accuracy {acc:.4f} -> {path}', flush=True)


def done(name):
    if os.path.exists(os.path.join(CKPT_DIR, name + '.pth')):
        print(f'[{name}] checkpoint exists, skipping', flush=True)
        return True
    return False


t0 = time.time()

# ---- victim ----
if not done('victim'):
    set_seed(0)
    victim_train = subset_dataset(0, train=True)
    loader = DataLoader(victim_train, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
    victim = fresh_resnet18()
    train_model(victim, loader, VICTIM_EPOCHS, lr=0.05, milestones=milestones_for(VICTIM_EPOCHS), log_prefix='[victim] ')
    save(victim, 'victim')

# ---- piracy models: fine-tune the victim on the attacker half ----
attacker_train = subset_dataset(1, train=True)
for i in range(N_PIRACY):
    if done(f'piracy_{i}'):
        continue
    set_seed(100 + i)
    sub = Subset(attacker_train, torch.randperm(len(attacker_train))[:15000].tolist())
    loader = DataLoader(sub, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
    net = fresh_resnet18()
    net.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'victim.pth'), map_location=DEVICE))
    train_model(net, loader, PIRACY_EPOCHS, lr=0.01, milestones=(99,), log_prefix=f'[piracy_{i}] ')
    save(net, f'piracy_{i}')

# ---- independent models: from scratch on the attacker half ----
for i in range(N_INDEP):
    if done(f'indep_{i}'):
        continue
    set_seed(200 + i)
    loader = DataLoader(attacker_train, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
    net = fresh_resnet18()
    train_model(net, loader, INDEP_EPOCHS, lr=0.05, milestones=milestones_for(INDEP_EPOCHS), log_prefix=f'[indep_{i}] ')
    save(net, f'indep_{i}')

print(f'total time {(time.time() - t0) / 60:.1f} min')
