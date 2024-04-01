
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
from utils import SubTrainDataset,compute_mean_std

#  [markdown]
# **Load Data**


os.environ['CUDA_VISIBLE_DEVICES'] = '3'  
device = "cuda:0"


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

#  [markdown]
# **Load Model**


fv = ms.get_model("resnet18",False, 10).to(device)
subset1_mean,subset2_mean = [0.49116027,0.49106753]
subset1_std, subset2_std = [0.24728487,0.24676652]
fv.load_state_dict(torch.load("/home/fazhong/studio/uap/checkpoint/fv_resnet18/Tuesday_19_March_2024_13h_44m_32s/training_session_1-98-best.pth"))# , map_location={'cuda:0':'cuda:3'}
fv = nn.Sequential(transforms.Normalize(subset1_mean, subset1_std), fv)

#  [markdown]
# **Embedding**


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


cifar10_subset1_loader = torch.utils.data.DataLoader(cifar10_subset1, batch_size=128, shuffle=False)
e,c,i = embedding(fv, cifar10_subset1_loader) 


i[1].shape

#  [markdown]
# **View data: output layer representation**


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
train_repre_data = torch.cat(e, dim=0)
train_repre_label = torch.cat(c, dim=0) 


X = train_repre_data.numpy()
y = train_repre_label.numpy()
print(X.shape)
tsne = TSNE(n_components=2, init='pca', random_state=0)
result = tsne.fit_transform(X)# use t-sne


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
x = result
for label in np.unique(y):
    position = ((y == label).reshape(-1))
    #此处position表示是否打印出来，True与False
    ax.scatter(x[position, 0], 
               x[position, 1], 
               label=label,
               s=1)
#     ax.set_xlabel('X[0]')
#     ax.set_ylabel('Y[0]')
# ax.legend(loc='best')
ax.set_title('', y=-0.2)
ax.legend(loc=0, bbox_to_anchor=(1.0, 1.0), borderaxespad=0)   

#plt.savefig('./visualize.pdf', bbox_inches = 'tight')#####################
#plt.show()

train_repre_data = torch.cat(e, dim=0)
train_label_data = torch.cat(c, dim=0)
train_image_data = torch.stack(i, dim=0)

train_image_data.shape

classes = train_label_data
imgs = train_image_data


cnt = 0
for count in range(train_repre_data.shape[0]):
    cnt += (np.argmax(train_repre_data[count]) == classes[count])
print(cnt/25000)


from sklearn.cluster import KMeans
kmodel = KMeans(n_clusters = 150)
kmodel.fit(train_repre_data)  


ls= kmodel.labels_

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
x = result
for label in range(50):
    position = (( ls == label).reshape(-1))
    #此处position表示是否打印出来，True与False
    ax.scatter(x[position, 0], 
               x[position, 1], 
               label=label,
               s=1)

#     ax.set_xlabel('X[0]')
#     ax.set_ylabel('Y[0]')
# ax.legend(loc='best')
ax.set_title('', y=-0.2)
#ax.legend(loc=0, bbox_to_anchor=(1.0, 1.0), borderaxespad=0)   

#plt.savefig('./visualize.pdf', bbox_inches = 'tight')#####################
#plt.show() 


from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=200, algorithm='ball_tree').fit(train_repre_data)
distances, indices = nbrs.kneighbors(train_repre_data) 

sets = []
for label in range(150):
    position = (( ls == label).reshape(-1))
    temp = np.arange(len(position))
    temp = temp[position]
    idx = np.random.choice(len(temp), 1)
    s = temp[idx]
    print(s[0], np.argmax(train_repre_data[s[0],:]) )
    print(classes[s])
    for i in s:
        cnt = 0
        knn = indices[i,:]
        print(knn.shape)
        for count in range(200):
            cnt += (classes[knn[count]] == classes[i])
        print(cnt)
        sets.append(knn) 


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
X = train_repre_data.numpy()
y = classes.numpy()
tsne = TSNE(n_components=2, init='pca', random_state=0)
result = tsne.fit_transform(X)# use t-sne 


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
x = result
for label, position in enumerate(sets):
    
    #此处position表示是否打印出来，True与False
    ax.scatter(x[position, 0], 
               x[position, 1], 
               label=label,
               s=1)

#     ax.set_xlabel('X[0]')
#     ax.set_ylabel('Y[0]')
# ax.legend(loc='best')
ax.set_title('', y=-0.2)
#ax.legend(loc=0, bbox_to_anchor=(1.0, 1.0), borderaxespad=0)   

#plt.savefig('./visualize.pdf', bbox_inches = 'tight')#####################
#plt.show()

data_sets = []
for s in sets:
    l = len(s) 
    dim = imgs[0].shape
    data_img = np.zeros((l,3,32,32))
    data_label = np.zeros((l,1))
    for i in range(l):
        data_img[i] = imgs[s[i]].reshape(1,3,32,32).numpy()
        data_label[i] = classes[s[i]].numpy()
    #print(data_img.shape)
    #print(data_label.shape)
    data_sets.append((data_img,data_label))


pickle.dump(data_sets, open("./result/split_sets_150points_200views.pkl", "wb"))

len(data_sets)

data_sets[1][1].shape




