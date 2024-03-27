""" helper function
author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch

from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
from PIL import Image


###########################
####### BEGIN NEW #########
###########################

class SubTrainDataset(Dataset):#需要继承data.Dataset
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)

# new function for load subdatasets (one is for victim and another is for the attacker)

def get_subtraining_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, sub_idx=1):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    X_set, y_set = pickle.load(open('./data/CIFAR100_sub_train_split.pkl', 'rb'))[sub_idx]
    
    cifar100_training = SubTrainDataset(X_set, list(y_set), transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

# new functions for the dataset **CIFAR10**

def get_subtraining_dataloader_cifar10(mean, std, batch_size=16, num_workers=2, shuffle=True, sub_idx=1):

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    X_set, y_set = pickle.load(open('./data/CIFAR10_sub_train_split.pkl', 'rb'))[sub_idx]
    
    cifar10_training = SubTrainDataset(X_set, list(y_set), transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

def get_training_dataloader_cifar10(mean, std, batch_size=16, num_workers=2, shuffle=True):

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar10_training = torchvision.datasets.CIFAR10(root='./data/CIFAR10/', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

def get_test_dataloader_cifar10(mean, std, batch_size=16, num_workers=2, shuffle=True):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar10_test = torchvision.datasets.CIFAR10(root='./data/CIFAR10/', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

# new function for load tiny imagenet network

def get_network_tinyimagenet(netname, gpu):
    if netname == 'vgg16':
        from torchvision.models import vgg16_bn
        net = vgg16_bn(num_classes=200)
    elif netname == 'vgg13':
        from torchvision.models import vgg13_bn
        net = vgg13_bn(num_classes=200)
    elif netname == 'densenet121':
        from torchvision.models import densenet121
        net = densenet121(num_classes=200)
    elif netname == 'resnet18':
        from torchvision.models import resnet18
        net = resnet18(num_classes=200)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if gpu: #use_gpu
        net = net.cuda()

    return net

def get_training_dataloader_tinyimagenet(mean, std, batch_size=16, num_workers=8, shuffle=True):

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tinyimagenet_training = datasets.ImageFolder('data/tiny-imagenet-200/train/', transform=transform_train)
    tinyimagenet_training_loader = DataLoader(
        tinyimagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return tinyimagenet_training_loader

def get_test_dataloader_tinyimagenet(mean, std, batch_size=16, num_workers=8, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    X_set, y_set = pickle.load(open('data/TinyImagenet_test.pkl', 'rb'))
    tinyimagenet_test = SubTrainDataset(X_set, list(y_set), transform=transform_test)
    tinyimagenet_test_loader = DataLoader(
        tinyimagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return tinyimagenet_test_loader

def get_subtraining_dataloader_tinyimagenet(mean, std, batch_size=16, num_workers=2, shuffle=True, sub_idx=1):

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    X_set, y_set = pickle.load(open('./data/TinyImagenet_sub_train_split.pkl', 'rb'))[sub_idx]
    
    tinyimagenet_training = SubTrainDataset(X_set, list(y_set), transform=transform_train)
    tinyimagenet_training_loader = DataLoader(
        tinyimagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return tinyimagenet_training_loader
###########################
######## END NEW ##########
###########################



def get_network(netname, gpu, num_classes=100):
    """ return given network
    """
    if netname == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif netname == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_classes=num_classes)
    elif netname == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes=num_classes)
    elif netname == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif netname == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(num_classes=num_classes)
    elif netname == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif netname == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif netname == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif netname == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif netname == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif netname == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif netname == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif netname == 'xception':
        from models.xception import xception
        net = xception()
    elif netname == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=num_classes)
    elif netname == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif netname == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif netname == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif netname == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif netname == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif netname == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif netname == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif netname == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif netname == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif netname == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif netname == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif netname == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif netname == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif netname == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif netname == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif netname == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif netname == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif netname == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif netname == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif netname == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif netname == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif netname == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif netname == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif netname == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif netname == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif netname == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif netname == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif netname == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif netname == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif netname == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./data/CIFAR100/', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='./data/CIFAR100/', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std




class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]




def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]





class SubTrainDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)

# new function for load subdatasets (one is for victim and another is for the attacker)




def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./data/CIFAR100/', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader



def compute_mean_std(dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([dataset[i][0][0, :, :] for i in range(len(dataset))])
    mean = numpy.mean(data_r)
    std = numpy.std(data_r)

    return mean, std

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./data/CIFAR100/', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='./data/CIFAR100/', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std




class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]




def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]



