
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
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

@torch.no_grad()
def embedding(net, data_loader):
    embeddings = []
    classes = []
    imgs = []
    net.eval()
    net.cpu()
    for (images, labels) in tqdm(data_loader):
        outputs = net(images)
        for i in range(images.shape[0]):
            embeddings.append(outputs[i].reshape(1,-1))
            classes.append(labels[i].reshape(1,-1))
            imgs.append(images[i])
    return embeddings, classes, imgs 

def fingerprintPointSelection(n_clusters, n_neighbors, net, data_loader):
    #@Params:
    #If your fingerprints consist of 200 datapoints and each fingerprint has 100 views, then n_cluster = 200 and n_neighbors = 100
    #The fingerprint Selection will base on the input network net and input data data_loader
    #@Return:
    #[(imgs of cluster1, labels of cluster1),..., (imgs of clustern, labels of clustern)]
    
    embedding, classes, imgs = embedding(net, data_loader)
    embedding = torch.cat(embedding, dim=0)
    classes = torch.cat(classes, dim=0)
    imgs = torch.cat(imgs, dim=0)
    
    kmodel = KMeans(n_clusters = n_clusters)
    kmodel.fit(embedding)
    ls= kmodel.labels_
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    #Choose one datapoints from each of the cluster as the generation point of this fingerprints and save it with its neighbors in list: sets
    sets = []
    for label in range(n_clusters):
        position = (( ls == label).reshape(-1))
        temp = np.arange(len(position))[position]
        idx = np.random.choice(len(temp), 1)
        s = temp[idx]
        for i in s:
            knn = indices[i,:]
            sets.append(knn)
    data_sets = []
    for s in sets:
        l = len(s) 
        dim = imgs[0].shape
        data_img = np.zeros((l,1,dim[0],dim[1]))
        data_label = np.zeros((l,1))
        for i in range(l):
            data_img[i] = imgs[s[i]].reshape(1,dim[0],dim[1]).numpy()
            data_label[i] = classes[s[i]].numpy()
        data_sets.append((data_img,data_label))
    pickle.dump(data_sets, open("./result/split_sets_" + str(n_clusters) + "_points_" + str(n_neighbors) + "views.pkl", "wb"))
    return data_sets



