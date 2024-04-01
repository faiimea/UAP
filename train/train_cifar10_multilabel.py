# train.py
#!/usr/bin/env	python3

""" train network using pytorch
author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_training_dataloader_cifar10, get_subtraining_dataloader_cifar10, get_test_dataloader_cifar10, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from model_structure import get_model


class MultiCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        #y:prediction x:groud-truth label
        softmax = nn.Softmax(dim = 1)
        x = softmax(x)
        y = softmax(y)
        y = torch.log(y)
        n = x.shape[0]
        l = x.shape[1]
        loss = 0
        for i in range(n):
            loss = loss - torch.dot(x[i],y[i])
        return 1/n*loss


# class MultiCrossEntropyLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.log_softmax = nn.LogSoftmax(dim=1)
        
#     def forward(self, inputs, targets):
#         # inputs: 网络输出（对数几率），targets: 真实标签（索引）
#         log_probs = self.log_softmax(inputs)
#         loss = nn.NLLLoss()(log_probs, targets)  # 使用NLLLoss来计算损失
#         return loss

def main(net, name, subset, gpu = True, b = 128, warm = 1, lr = 0.1, resume = False, newloader = None, 
         save_path = "./checkpoint/", normalize = False, device =  torch.device("cuda:0"),
        epoch = 100, test_dataloader = None, CIFAR10_MILESTONES = [40, 70]):
    def train(epoch):

        start = time.time()
        net.train()
        for batch_index, (images, labels) in enumerate(cifar10_training_loader):

            if gpu:
                labels = labels.to(device)
                images = images.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(cifar10_training_loader) + batch_index + 1

            last_layer = list(net.children())[-1]
        
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * b + len(images),
                total_samples=len(cifar10_training_loader.dataset)
            ))


            if epoch <= warm:
                warmup_scheduler.step()

        finish = time.time()

        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    @torch.no_grad()
    def eval_training(epoch=0, tb=True):

        start = time.time()
        net.eval()

        test_loss = 0.0 # cost function error
        correct = 0.0

        for (images, labels) in cifar10_test_loader:

            if gpu:
                images = images.to(device)
                labels = labels.to(device)

            outputs = net(images)
            loss = test_loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        finish = time.time()
        #if gpu:
           # print('GPU INFO.....')
           #print(torch.cuda.memory_summary(), end='')
        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(cifar10_test_loader.dataset),
            correct.float() / len(cifar10_test_loader.dataset),
            finish - start
        ))
        print()

        #add informations to tensorboard
        return correct.float() / len(cifar10_test_loader.dataset)
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-subset', type=int, default=None, help='subset index, 1 or 2')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()
    """
    CIFAR10_SUBTRAIN_MEAN = [0.49116027,0.49106753]
    CIFAR10_SUBTRAIN_STD = [0.24728487,0.24676652]
    CIFAR10_MILESTONES = CIFAR10_MILESTONES 
    CHECKPOINT_PATH = save_path
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    LOG_DIR = 'runs'
    SAVE_EPOCH = 10
    EPOCH = epoch
    
    if(isinstance(net,str)):
        net = get_model(net,10)
    net = net.to(device)
    
    if subset < 3:
        set_mean = CIFAR10_SUBTRAIN_MEAN[subset - 1]
        set_std = CIFAR10_SUBTRAIN_STD[subset - 1]
        cifar10_training_loader = get_subtraining_dataloader_cifar10(
            set_mean,
            set_std,
            num_workers=8,
            batch_size=b,
            shuffle=True,
            sub_idx=subset - 1
        )
    else: 
        set_mean = 0
        set_std = 1 
        cifar10_training_loader = newloader
    chkfolder_subset = name

    if normalize == True:
        cifar10_test_loader = get_test_dataloader_cifar10(
            set_mean,
            set_std,
            num_workers=8,
            batch_size=b,
            shuffle=False
        )
    else:
        cifar10_test_loader = get_test_dataloader_cifar10(
            0,
            1,
            num_workers=8,
            batch_size=b,
            shuffle=False
        )
    if test_dataloader != None:
        cifar10_test_loader = test_dataloader


    loss_function = MultiCrossEntropyLoss()
    test_loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum = 0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=CIFAR10_MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar10_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    if resume:
        recent_folder = most_recent_folder(os.path.join(CHECKPOINT_PATH,chkfolder_subset), fmt=DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')
        checkpoint_path = os.path.join(CHECKPOINT_PATH, chkfolder_subset, recent_folder)
    else:
        checkpoint_path = os.path.join(CHECKPOINT_PATH, chkfolder_subset, TIME_NOW)

    #use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)


    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{name}-{epoch}-{type}.pth')

    best_acc = 0.0
    if resume:
        best_weights = best_acc_weights(os.path.join(CHECKPOINT_PATH, chkfolder_subset, recent_folder))
        if best_weights:
            weights_path = os.path.join(CHECKPOINT_PATH, chkfolder_subset, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))
        recent_weights_file = most_recent_weights(os.path.join(CHECKPOINT_PATH, chkfolder_subset, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(CHECKPOINT_PATH, chkfolder_subset, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(CHECKPOINT_PATH, chkfolder_subset, recent_folder))


    for epoch in range(1,  EPOCH + 1):
        if epoch > warm:
            train_scheduler.step(epoch)

        if resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > CIFAR10_MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(name=name, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % SAVE_EPOCH:
            weights_path = checkpoint_path.format(name=name, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

if __name__ =='__main__':
    print('?')
    net_name = 'resnet34'  # Specify the name of the network you want to use.
    session_name = 'fv_resnet34'  # Give a name to this training session.
    subset_index = 2  # Choose a subset index, for example, 1.
    main(net_name,session_name,subset_index)