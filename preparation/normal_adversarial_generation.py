
os.environ['CUDA_VISIBLE_DEVICES']='1'

import os
import random
import numpy as np
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import SubTrainDataset,compute_mean_std
import uap.train.model_structure as ms

from uap.train.train_cifar10 import *
from art.attacks.evasion import FastGradientMethod, UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

torch.cuda.set_device(0)

Set1, Set2 = pickle.load(open('./data/CIFAR10_sub_train_split.pkl', 'rb'))
transform_train1 = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
#         transforms.Normalize(settings.CIFAR10_SUBTRAIN_MEAN[0], settings.CIFAR10_SUBTRAIN_STD[0])
    ])

X_set1, y_set1 = Set1

cifar10_subset1 = SubTrainDataset(X_set1, list(y_set1), transform=transform_train1)

transform_train2 = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
#         transforms.Normalize(settings.CIFAR10_SUBTRAIN_MEAN[1], settings.CIFAR10_SUBTRAIN_STD[1])
    ])

X_set2, y_set2 = Set2

cifar10_subset2 = SubTrainDataset(X_set2, list(y_set2), transform=transform_train2)
data_dir = os.path.join("./data", 'CIFAR10')
transform_test = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
#         transforms.Normalize(settings.CIFAR10_SUBTRAIN_MEAN[0], settings.CIFAR10_SUBTRAIN_STD[0])
    ])
cifar10_testset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_test)
cifar10_subset1_loader = torch.utils.data.DataLoader(cifar10_subset1, batch_size=128, shuffle=True)
cifar10_subset2_loader = torch.utils.data.DataLoader(cifar10_subset2, batch_size=128, shuffle=True)
cifar10_test_loader = torch.utils.data.DataLoader(cifar10_testset, batch_size=128, shuffle=False)


fv = get_model("resnet18",True, 10)
subset1_mean,subset2_mean = [0.49116027,0.49106753]
subset1_std, subset2_std = [0.24728487,0.24676652]
fv.load_state_dict(torch.load("./checkpoint/fv_resnet18/Tuesday_19_October_2021_07h_01m_19s/fv_resnet18-84-best.pth"))
fv = nn.Sequential(transforms.Normalize(subset1_mean, subset1_std), fv)
                   
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fv.parameters(), lr=0.001)
classifier = PyTorchClassifier(
    model=fv,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
    device_type = 'gpu'
)


sets = pickle.load(open("./result/split_sets_150points_200views.pkl", 'rb'))
nb_p = sets[0][0].shape[0]
nb_c = len(sets)
new_set_image = [ torch.zeros((nb_c,3,32,32)) for i in range(nb_p)]
new_set_label = [ torch.zeros((nb_c)) for i in range(nb_p)]
new_datasets = []
for cnt,s in enumerate(sets):
    for i in range(nb_p):
        new_set_image[i][cnt] = torch.from_numpy(s[0][i])
        new_set_label[i][cnt] = torch.from_numpy(s[1][i])
for i in range(nb_p):
    new_datasets.append(torch.utils.data.TensorDataset(new_set_image[i],new_set_label[i]))


from art.attacks.evasion import DeepFool


def deepfool(Fmnist_subset1_loader, name):
    attacker = DeepFool(classifier)
    for cnt, (images, labels) in enumerate (Fmnist_subset1_loader):
        adv = attacker.generate(x=images) 
        print(adv.shape)
        if not os.path.exists("./result/normal_adv/review/" + str(name) + "/"):
            os.makedirs("./result/normal_adv/review/" + str(name) + "/")
        pickle.dump(images, open("./result/normal_adv/review/" + str(name) + "/" +  str(cnt) + "_adv.pkl", "wb"))
        pickle.dump(adv, open("./result/normal_adv/review/" + str(name) +  "/" + str(cnt) + "_clean.pkl", "wb"))


deepfool(cifar10_subset1_loader, 1)


def generate_datapoint(net, sets, b_z = 32, nb_class = 10):
    """
    param:
    net: f
    data_loader: data points used as queries
    pert: UAP
    label: y
    b_z: batch_size of return
    nb_class: the number of classification class

    return:
    (xi,yi) as an dataloader
    """
    cnt = 0
    print(cnt)
    for data_set in sets:
        cnt += 1
        data_loader = torch.utils.data.DataLoader(data_set, b_z)
        deepfool(data_loader,cnt)


generate_datapoint(fv, new_datasets )


