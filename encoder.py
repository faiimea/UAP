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
import torch.nn.functional as F
import os

class classifier(nn.Module):
        def __init__(self, input_channel):
                super(classifier, self).__init__()
                self.fc1 = nn.Linear(input_channel, 50)
                self.fc2 = nn.Linear(50,20) 
                self.fc3 = nn.Linear(20, 4)

        def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
            
        def feature(self,x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return x



