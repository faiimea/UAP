
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




import random

# torch.cuda.set_device(0)
# device = torch.device("cuda:0")
# fv = ms.get_model("resnet18",True, 10)
# subset1_mean,subset2_mean = [0.49116027,0.49106753]
# subset1_std, subset2_std = [0.24728487,0.24676652]
# fv.load_state_dict(torch.load("./checkpoint/fv_resnet18/Tuesday_19_October_2021_07h_01m_19s/fv_resnet18-84-best.pth")) 
# fv = nn.Sequential(transforms.Normalize(subset1_mean, subset1_std), fv)


# cifar10_training_loader1 = get_subtraining_dataloader_cifar10(
#         0,
#         1,
#         num_workers=6,
#         batch_size=128,
#         shuffle=True,
#         sub_idx=1
#     )


# c = []
# for cnt in range(10):
#         print("=====================model extraction " + str(cnt) + "==============================")
#         extraction_model = ms.get_model("resnet34", True, 10)
#         print(extraction_model)
#         #extraction_model.load_state_dict(torch.load("./result/struc1-pretrain-6-regular.pth"))
#         extraction_model = nn.Sequential(transforms.Normalize(subset1_mean, subset1_std),extraction_model)
#         extraction_model = simple_model_extraction(fv, extraction_model,cifar10_training_loader1, 
#                                                     "./checkpoint/simple_softlabel/",  lr = 0.0001, MAX_ITER=500)     


