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
from torch.utils.data import TensorDataset


def concanate_fingerprint(net, data_sets, pert, label, b_z = 32, nb_class = 10):
    """
    param:
    net: f
    data_sets: [(imgs of cluster1, labels of cluster1),..., (imgs of clustern, labels of clustern)] 
    pert: UAP vector
    label: 0 for victim's fingerprint, 1 for homologous fingerprints, 2 for independent fingerprints
    b_z: batch_size of return
    nb_class: the number of classification class

    return:
    [fingerprint_view_0,..., fingerprint_view_k]
    """
    

    # n_clusters = 200

    
    data_sets = regroup_datapoints(data_sets)
    n_clusters = len(data_sets)
    n_neighbors = len(data_sets[0])
    features = torch.zeros( n_clusters, 2 * nb_class * n_neighbors)
    # print(features.shape)
    label = (torch.ones(n_clusters) * label).type(torch.long)
    softmax = nn.Softmax(dim=1)
    
    cnt=0
    for data_set in data_sets:
        # print(cnt)      
        data_loader = torch.utils.data.DataLoader(data_set, b_z)
        batch_size = b_z
        n = len(data_set)
        #print(n)
        #n=200


        # total_num=0
        # for i, (inputs, labels) in enumerate(data_loader, 0):
        #     total_num+=1
        # print(total_num)
        for i, (inputs, labels) in enumerate(data_loader, 0):
            start = batch_size * i
            end = min(batch_size * (i+1), n)
            #print(start,end)
            net.eval()
            with torch.no_grad():
                inputs, labels = inputs.cuda(), labels.cuda()
                per_inputs = (inputs + torch.tensor(pert).cuda()).cuda()
                outputs = softmax(net(inputs).cpu())
                #print(outputs.shape)
                pert_outputs = softmax(net(per_inputs).cpu())
                
                o = torch.cat((outputs, pert_outputs), 1).reshape(-1)
                # print(o.shape)
                #print(features.shape)
                # print(start*20, end*20)
                
                features[cnt,start*2*nb_class:end*2*nb_class] = o
                
                #print(features)
        cnt+=1
                
    new_dataset = torch.utils.data.TensorDataset(features,label)
    data_loader = torch.utils.data.DataLoader(new_dataset, b_z, shuffle=False)
    return data_loader 

def regroup_datapoints(data_sets):
    #@Param:
    #data_sets: [(imgs of cluster1, labels of cluster1),..., (imgs of clustern, labels of clustern)] 
    #@Return:
    #new_data_sets: [(img of view1, labels of view1),..., (imgs of viewn, labels of viewn)] 
    
    n_neighbors = data_sets[0][0].shape[0]
    n_clusters = len(data_sets)
    # Here (3,32,32) is of CIFAR 10
    new_set_image = [torch.zeros((n_clusters,3,32,32)) for i in range(n_neighbors)]
    new_set_label = [torch.zeros((n_clusters)) for i in range(n_neighbors)]
    new_data_sets = []
    
    for cnt,s in enumerate(data_sets):
        for i in range(n_neighbors):
            new_set_image[i][cnt] = torch.from_numpy(s[0][i])
            new_set_label[i][cnt] = torch.from_numpy(s[1][i])
    for i in range(n_neighbors):
        new_data_sets.append(torch.utils.data.TensorDataset(new_set_image[i],new_set_label[i]))
    return new_data_sets
# A list with len of n_neighbor and one element has n_cluster images


