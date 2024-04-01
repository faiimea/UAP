
# Homo models

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

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
from utils import * 
import model_structure as ms
import uap.train.train_cifar10 as train_cifar10  

data_dir = os.path.join("./data", 'CIFAR10')
transform_test = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
#         transforms.Normalize(settings.CIFAR10_SUBTRAIN_MEAN[0], settings.CIFAR10_SUBTRAIN_STD[0])
    ])
cifar10_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_test) 

for i in range(0,10):
    print("=====================model training " + str(i) + "==============================")
    train_size = int(0.4 * len(cifar10_trainset))  
    sub_train_size = len(cifar10_trainset) - train_size
    train_dataset, sub_train_dataset = torch.utils.data.random_split(cifar10_trainset, [train_size, sub_train_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, 64, shuffle = True)
    
    
    train_cifar10.main("resnet18", "resnet18_indep_n", 3, newloader = train_loader,epoch = 80, device = "cuda:0",CIFAR10_MILESTONES=[40,60])                


