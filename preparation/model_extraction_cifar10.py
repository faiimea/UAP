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
import torch.utils.data as udata
from model_structure import get_model
import pickle
from utils import get_training_dataloader_cifar10, get_subtraining_dataloader_cifar10, get_test_dataloader_cifar10, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


torch.cuda.set_device(1)

CIFAR10_SUBTRAIN_MEAN = [0.49116027,0.49106753]
CIFAR10_SUBTRAIN_STD = [0.24728487,0.24676652]
set_mean = CIFAR10_SUBTRAIN_MEAN[1]
set_std = CIFAR10_SUBTRAIN_STD[1]
cifar10_training_loader = get_subtraining_dataloader_cifar10(
        set_mean,
        set_std,
        num_workers=8,
        batch_size=128,
        shuffle=True,
        sub_idx=1
    )
cifar10_test_loader = get_test_dataloader_cifar10(
        set_mean,
        set_std,
        num_workers=8,
        batch_size=128,
        shuffle=False
    )


def augment_dataset(model, dataset, lr = 0.01):

    new_dataset = list()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for img, label in data_loader:

        img, label = Variable(img.cuda(), requires_grad=True), Variable(label.cuda())
        _, label = torch.max(label, 1)
        model.eval()
        output = model(img)
        #print(output.shape)
        output[0][label].backward()

        img_new = img[0] + lr * torch.sign(img.grad.data[0])
        img.grad.data.zero_()

        #print(img.grad.data.shape)
        new_dataset.append(img_new.cpu())
        

    new_dataset = torch.stack([data_point for data_point in new_dataset])
    new_dataset = udata.TensorDataset(new_dataset)

    return new_dataset


def label_dataset(net, sub_trainloader, initial_label = False, nb_class = 10):
    softmax = nn.Softmax()
    shape = sub_trainloader.dataset[0][0].shape
    sub_trainimg = torch.zeros(len(sub_trainloader.dataset),shape[0],shape[1],shape[2])
    sub_trainlabel = torch.zeros((len(sub_trainloader.dataset),nb_class))
    batch_size = sub_trainloader.batch_size
    if(initial_label == True):
        for i, (inputs, labels) in enumerate(sub_trainloader, 0):
            start = batch_size * i
            end = min(batch_size * (i+1), len(sub_trainloader.dataset))
            net.eval()
            with torch.no_grad():
                # Transfer to GPU
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                predicted = softmax(outputs)
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
                inputs = inputs.cuda()
                outputs = net(inputs)
                predicted = softmax(outputs)
                sub_trainlabel[start:end] = predicted.cpu()
                sub_trainimg[start:end] = inputs.cpu()
            
    return sub_trainimg,sub_trainlabel


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

torch.cuda.set_device(1)



def main(net, name, trainloader, testloader, EPOCH = 100, gpu = True, b = 128, warm = 1, lr = 0.1, resume = False):
    def train(epoch):

        start = time.time()
        net.train()
        for batch_index, (images, labels) in enumerate(cifar10_training_loader):

            if gpu:
                labels = labels.cuda()
                images = images.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            predicted = softmax(outputs)
            loss = loss_function(predicted, labels)
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(cifar10_training_loader) + batch_index + 1

            last_layer = list(net.children())[-1]
            for name, para in last_layer.named_parameters():
                if 'weight' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * b + len(images),
                total_samples=len(cifar10_training_loader.dataset)
            ))

            #update training loss for each iteration
            writer.add_scalar('Train/loss', loss.item(), n_iter)

            if epoch <= warm:
                warmup_scheduler.step()

        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

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
                images = images.cuda()
                labels = labels.cuda()

            outputs = net(images)
            predicted = softmax(outputs)
            loss = loss_function(predicted, labels)

            test_loss += loss.item()
            _, preds = predicted.max(1)
            _, labels = labels.max(1)
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
        if tb:
            writer.add_scalar('Test/Average loss', test_loss / len(cifar10_test_loader.dataset), epoch)
            writer.add_scalar('Test/Accuracy', correct.float() / len(cifar10_test_loader.dataset), epoch)

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
    CIFAR10_MILESTONES = [40, 70]
    CHECKPOINT_PATH = "./checkpoint/"
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    LOG_DIR = 'runs'
    SAVE_EPOCH = 10
    
    net = net.cuda()
    cifar10_training_loader = trainloader
    chkfolder_subset = name

    
    cifar10_test_loader = testloader

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=CIFAR10_MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar10_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
    softmax = nn.Softmax()
    
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

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            LOG_DIR, name, TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

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

    writer.close()


def train_substitute(oracle_model, extraction_model, train_loader, test_loader,name, MAX_ROUND = 3, EPOCHS = [10,30,50,70], LAMBDA = 0.01, n_class = 10):
    dataset = train_loader.dataset
    oracle_model = oracle_model.cuda()
    input_shape = list(dataset[0][0].shape)
    model = extraction_model.cuda()

    dummy_img, dummy_label = label_dataset(oracle_model, train_loader, True)
    dummy_dataset = torch.utils.data.TensorDataset(dummy_img,dummy_label)
    dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=128, shuffle=True)
    
    dummy_testimg, dummy_testlabel = label_dataset(oracle_model, test_loader,True)
    dummy_testdataset = torch.utils.data.TensorDataset(dummy_testimg,dummy_testlabel)
    dummy_testdataloader = torch.utils.data.DataLoader(dummy_testdataset, batch_size=128, shuffle=True)

    for round in range(MAX_ROUND):
        print("===round:",round,"start===")
        input_shape = list(dataset[0][0].shape)
        
        #print("start training with substitute data")
       
        print("Train the model with ",len(dummy_dataloader.dataset),"data")
        main(model, name , dummy_dataloader,dummy_testdataloader,EPOCHS[round])

        #print("data augmentation process begins")
        if(round == MAX_ROUND - 1):
            continue
        dataset = augment_dataset(model, dummy_dataset, LAMBDA)

        new_train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        new_dummy_img, new_dummy_label = label_dataset(oracle_model, new_train_loader)
        new_dummy_dataset = torch.utils.data.TensorDataset(new_dummy_img,new_dummy_label)
        dummy_dataset = udata.ConcatDataset([dummy_dataset, new_dummy_dataset])
        #print("label substitute training dataset finished")
        dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=128, shuffle=True)
        


def extract_without_augmentation(oracle_model, extraction_model, train_loader, test_loader, name, EPOCHS = 100, n_class = 10, resume = False):
    dataset = train_loader.dataset
    oracle_model = oracle_model.cuda()
    input_shape = list(dataset[0][0].shape)
    model = extraction_model.cuda()

    dummy_img, dummy_label = label_dataset(oracle_model, train_loader, True)
    dummy_dataset = torch.utils.data.TensorDataset(dummy_img,dummy_label)
    dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=128, shuffle=True)
    
    dummy_testimg, dummy_testlabel = label_dataset(oracle_model, test_loader,True)
    dummy_testdataset = torch.utils.data.TensorDataset(dummy_testimg,dummy_testlabel)
    dummy_testdataloader = torch.utils.data.DataLoader(dummy_testdataset, batch_size=128, shuffle=True)
    
    main(model, name , dummy_dataloader, dummy_testdataloader, EPOCHS, resume = resume)


p = 0.8
weights_path = "/home/fazhong/studio/uap/checkpoint/fv_resnet18/Tuesday_19_March_2024_13h_44m_32s/training_session_1-98-best.pth"
fv = get_model("resnet18",10)
fv.load_state_dict(torch.load(weights_path))


extraction_model = get_model("resnet18",10)
data_set = cifar10_training_loader.dataset
train_size = int(p * len(data_set))
sub_train_size = len(data_set) - train_size
train_dataset, sub_train_dataset = torch.utils.data.random_split(data_set, [train_size, sub_train_size])
sub_train_loader = torch.utils.data.DataLoader(sub_train_dataset, batch_size=128, shuffle=True)


extract_without_augmentation(fv, extraction_model, cifar10_training_loader, cifar10_test_loader, "resnet18_ext_sub1")


extraction_model, acc = train_substitute(fv, extraction_model, sub_train_loader, cifar10_test_loader, "resnet18_extract",MAX_ROUND = 4)


