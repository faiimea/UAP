import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import torch.nn as nn
from torch import optim
import copy
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

class Contrastiveloss(nn.Module):
    def __init__(self, tau = 0.1):
        self.tau = tau
        super().__init__()
    
    def similarity(self, x1 , x2):
        return (x1.T @ x2) / (torch.norm(x1,2) * torch.norm(x2,2))
    
    # Change loss here!
    def single_image_loss(self, x1, y1, x, y,fp_v):
        same = torch.where(y1 == y)[0]
        diff = torch.where(y1 != y)[0]
        top = 0.0
        down = 0.0
        for i in same:
            top += torch.exp(self.similarity(x1, x[i]) / self.tau)
        for i in range(x.shape[0]):
            down += torch.exp(self.similarity(x1, x[i]) / self.tau)

        # 0?
        # print(y)
        # print(y1)
        # if y1 == 2:
        #     top += torch.exp(self.similarity(x1, fp_v) / self.tau)
        # # elif y1 == 0:
        # #      top += torch.exp(self.similarity(x1, fp_v) *2 / self.tau)
        # elif y1 == 1:
        #     down += torch.exp(self.similarity(x1, fp_v) / self.tau)
        # if(top == 0 or down == 0):
        #     for i in same:
        #         top += torch.exp(self.similarity(x1, x[i]) / self.tau)
        #     for i in diff:
        #         down += torch.exp(self.similarity(x1, x[i]) / self.tau)
        return -torch.log(top/down) 
    
    def forward(self, x, y,fp_v):
        loss = 0
        for i in range(x.shape[0]):
            loss += self.single_image_loss(x[i],y[i],x,y,fp_v)
            
        return 1/(x.shape[0])*loss

    
def train(net,trainloader,testloader,n_epochs,optimizer, fp_v, tau=0.2, device = "cuda:0"):
        accuracy = 0
        net.train()

        # print(testloader.shape)
        # print(fp_v[0].shape)
        # print(fp_v)
        # for inputs, labels in testloader:
            # print(inputs.shape)
            # idx_indep = (labels == 2)
            # print(inputs[idx_indep,:])
            # fp_indep = net.feature(inputs[idx_indep,:])
            # for cnt in range(fp_indep.shape[0]):
        fp_v = tuple(item.to(device) for item in fp_v)
        for epoch in range(n_epochs):
                print(epoch)
                running_loss = 0.0
                for i, (inputs, labels) in enumerate(trainloader, 0):
                        # Transfer to GPU
                        # if i == 1: break
                        inputs, labels = inputs.to(device), labels.to(device)
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward + backward + optimize
                        outputs = net(inputs)
                        contras = Contrastiveloss(tau)
                        
                        loss = contras(net.feature(inputs),labels,net.feature(fp_v[0]))            
                        print("loss:",loss)
                        loss.backward()
                        optimizer.step()
                evaluate(net, fp_v, testloader, device)
        print('Finished Training')

        # 指定保存模型的路径
        model_save_path = '/home/fazhong/studio/uap/result/model.pth'
        torch.save(net.state_dict(), model_save_path)


        return net

    
def similarity(x1 , x2):
    return (x1 @ x2.T) / (torch.norm(x1,2) * torch.norm(x2,2))

def evaluate(net, fp_v, test_loader, device = "cuda:0"):
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
    net = net.to(device)
    sim_indep = 0
    sim_homo = 0
    cnt_indep = 0
    cnt_homo = 0
    if isinstance(fp_v, tuple):
         fp_v = tuple(item.to(device) for item in fp_v)
    # print(fp_v[0].shape)
    # print(fp_v[0][1])
    fp_v_net=net.feature(fp_v[0])

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        idx_indep = (labels == 2)
        idx_homo = (labels == 1)
        fp_indep = net.feature(inputs[idx_indep,:])
        # print(inputs[idx_indep,:].shape)
        # print(inputs[idx_indep,:][1])
        for cnt in range(fp_indep.shape[0]):
            sim_indep += similarity(fp_v_net, fp_indep[cnt])
            # cnt_indep += fp_indep.shape[0]
            cnt_indep+=1
        fp_homo = net.feature(inputs[idx_homo,:])
        for cnt in range(fp_homo.shape[0]):
            sim_homo += similarity(fp_v_net, fp_homo[cnt])
            #cnt_homo += fp_homo.shape[0]
            cnt_homo+=1
            
    print("similarity of homologous models:",sim_homo/cnt_homo)
    print("similarity of independent models:",sim_indep/cnt_indep)
    
# def evaluate(net, fp_v, test_loader, device="cuda:0"):
#     net.to(device)  # 将网络移动到指定设备
#     if isinstance(fp_v, tuple):
#         fp_v = tuple(item.to(device) for item in fp_v)  # 如果fp_v是元组中的张量，将它们移动到设备
#     else:
#         fp_v = fp_v.to(device)  # 如果fp_v是单个张量，直接移动到设备
    
#     sim_indep = 0
#     sim_homo = 0
#     cnt_indep = 0
#     cnt_homo = 0
    
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)  # 确保输入和标签也在指定的设备上
#         idx_indep = (labels == 2).to(device)  # 确保索引也在指定的设备上
#         idx_homo = (labels == 1).to(device)  # 确保索引也在指定的设备上

#         if idx_indep.any():
#             fp_indep = net.feature(inputs[idx_indep,:])
#             for cnt in range(fp_indep.shape[0]):
#                 sim_indep += similarity(fp_v, fp_indep[cnt])
#             cnt_indep += fp_indep.shape[0]

#         if idx_homo.any():
#             fp_homo = net.feature(inputs[idx_homo])
#             for cnt in range(fp_homo.shape[0]):
#                 sim_homo += similarity(fp_v, fp_homo[cnt])
#             cnt_homo += fp_homo.shape[0]

#     if cnt_homo > 0:
#         print("similarity of homologous models:", sim_homo / cnt_homo)
#     else:
#         print("No homologous models to evaluate.")

#     if cnt_indep > 0:
#         print("similarity of independent models:", sim_indep / cnt_indep)
#     else:
#         print("No independent models to evaluate.")

