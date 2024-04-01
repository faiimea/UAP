
import os
import numpy as np
import torchvision
from tqdm import tqdm
import pickle
np.random.seed(0)


dataset = "CIFAR10"
data_dir = os.path.join("./data/", dataset)
print('Data stored in %s' % data_dir)
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
vic_num = len(trainset) // 2
vic_idx = np.random.choice(len(trainset), vic_num, replace=False)
print(vic_idx)


testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=None)


X_list = []
y_list = []
for data in tqdm(trainset):
    x,y = data
    X_list.append(np.array(x))
    y_list.append(y)
X_np = np.array(X_list)
y_np = np.array(y_list)
X_set1 = X_np[vic_idx]
y_set1 = y_np[vic_idx]
X_set2 = X_np[~vic_idx]
y_set2 = y_np[~vic_idx]
pickle.dump(((X_set1, y_set1), (X_set2, y_set2)), open("./data/{}_sub_train_split.pkl".format("CIFAR10"), "wb"))


