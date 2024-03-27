
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 



import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torch as torch
import copy
#from torch.autograd.gradcheck import zero_gradients
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import model_structure as ms
from utils import *
import train_cifar10 as train 
import train_cifar10_multilabel as train_cifar10 

def label_dataset_onelabel(net, sub_trainloader, initial_label = False, nb_class = 10, device = torch.device("cuda:0")):
    softmax = nn.Softmax()
    shape = sub_trainloader.dataset[0][0].shape
    sub_trainimg = torch.zeros(len(sub_trainloader.dataset),shape[0],shape[1],shape[2])
    sub_trainlabel = torch.zeros(len(sub_trainloader.dataset)).long()
    batch_size = sub_trainloader.batch_size
    if(initial_label == True):
        for i, (inputs, labels) in enumerate(sub_trainloader, 0):
            start = batch_size * i
            end = min(batch_size * (i+1), len(sub_trainloader.dataset))
            net.eval()
            with torch.no_grad():
                # Transfer to GPU
                #print(inputs)
                #printnt(labels)
                inputs, labels = inputs.to(device), labels.to(device)
                met = net.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                sub_trainlabel[start:end] = predicted.cpu()
                sub_trainimg[start:end] = inputs.cpu()

    else:
        for i, inputs in enumerate(sub_trainloader, 0):
            inputs = inputs[0]
            start = batch_size * i
            end = min(batch_size * (i+1), len(sub_trainloader.dataset))
            net.eval()
            with torch.no_grad():
                # Transfer to GPU
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                sub_trainlabel[start:end] = predicted.cpu()
                sub_trainimg[start:end] = inputs.cpu()
            
    return sub_trainimg, sub_trainlabel

def label_dataset_softlabel(net, sub_trainloader, initial_label = False, nb_class = 10,device = torch.device("cuda:0") ):
    softmax = nn.Softmax()
    shape = sub_trainloader.dataset[0][0].shape
    sub_trainimg = torch.zeros(len(sub_trainloader.dataset),shape[0],shape[1],shape[2])
    sub_trainlabel = torch.zeros((len(sub_trainloader.dataset),nb_class))
    batch_size = sub_trainloader.batch_size
    if(initial_label == True):
        for i, (inputs, labels) in enumerate(sub_trainloader, 0):
            start = batch_size * i
            end = min(batch_size * (i+1), len(sub_trainloader.dataset))
            net = net.to(device)
            net.eval()
            with torch.no_grad():
                # Transfer to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                predicted = net(inputs)
                #predicted = softmax(outputs)
                sub_trainlabel[start:end] = predicted.cpu()
                sub_trainimg[start:end] = inputs.cpu()

    else:
        for i, inputs in enumerate(sub_trainloader, 0):
            inputs = inputs[0]
            start = batch_size * i
            end = min(batch_size * (i+1), len(sub_trainloader.dataset))
            net.eval()
            with torch.no_grad():
                # Transfer to GPU
                inputs = inputs.to(device)
                predicted = net(inputs)
                #predicted = softmax(outputs)
                sub_trainlabel[start:end] = predicted.cpu()
                sub_trainimg[start:end] = inputs.cpu()
            
    return sub_trainimg,sub_trainlabel

def simple_model_extraction(oracle_model, ext_model, train_loader,save_path, MAX_ITER = 100,batch_size = 128,  lr = 0.01): 
    test_dataloader = get_test_dataloader_cifar10(
            0,
            1,
            num_workers=8,
            batch_size=128,
            shuffle=False
        )
    test_dummy_img, test_dummy_label = label_dataset_onelabel(oracle_model, test_dataloader,True)
    test_dummy_dataset = torch.utils.data.TensorDataset(test_dummy_img,test_dummy_label)
    test_dummy_dataloader = torch.utils.data.DataLoader(test_dummy_dataset, batch_size=128, shuffle=True)
    new_img, new_label = label_dataset_softlabel(oracle_model, sub_trainloader = train_loader, initial_label = True, nb_class = 10, device = "cuda:0")
    new_dataset = torch.utils.data.TensorDataset(new_img,new_label)
    new_dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
    train_cifar10.main(ext_model, "simple_onelabel_resnet34", 3, newloader = new_dataloader, b = batch_size, 
                       save_path = save_path, normalize = True, epoch = MAX_ITER, test_dataloader = test_dummy_dataloader,
                       lr = lr
                )


import random
torch.cuda.set_device(0)
device = torch.device("cuda:0")
fv = ms.get_model("resnet18",True, 10)
subset1_mean,subset2_mean = [0.49116027,0.49106753]
subset1_std, subset2_std = [0.24728487,0.24676652]
fv.load_state_dict(torch.load("/home/fazhong/studio/uap/checkpoint/training_session_1/Tuesday_19_March_2024_14h_22m_09s/training_session_1-87-best.pth")) 
fv = nn.Sequential(transforms.Normalize(subset1_mean, subset1_std), fv)


cifar10_training_loader1 = get_subtraining_dataloader_cifar10(
        0,
        1,
        num_workers=6,
        batch_size=128,
        shuffle=True,
        sub_idx=1
    )


c = []
for cnt in range(10):
        print("=====================model extraction " + str(cnt) + "==============================")
        extraction_model = ms.get_model("resnet18", True, 10)
        print(extraction_model)
        #extraction_model.load_state_dict(torch.load("./result/struc1-pretrain-6-regular.pth"))
        extraction_model = nn.Sequential(transforms.Normalize(subset1_mean, subset1_std),extraction_model)
        extraction_model = simple_model_extraction(fv, extraction_model,cifar10_training_loader1, 
                                                    "./checkpoint/simple_softlabel/",  lr = 0.0001, MAX_ITER=500)     


