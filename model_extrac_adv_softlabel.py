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
from utils import *
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from art.attacks.evasion import CarliniL2Method
import train_cifar10_multilabel as train_cifar10
import train_cifar10 as train


import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)


def label_dataset_softlabel(net, sub_trainloader, initial_label = False, nb_class = 10,device = torch.device("cuda:0") ):
    softmax = nn.Softmax()
    shape = sub_trainloader.dataset[0][0].shape
    sub_trainimg = torch.zeros(len(sub_trainloader.dataset),shape[0],shape[1],shape[2])
    sub_trainlabel = torch.zeros((len(sub_trainloader.dataset),nb_class))
    batch_size = sub_trainloader.batch_size
    if(initial_label == True):
        for i, (inputs, labels) in enumerate(sub_trainloader, 0):
            start = batch_size * i
            end = min(batch_size * (i+1), len(sub_trainloader.dataset))
            net = net.to(device)
            net.eval()
            with torch.no_grad():
                # Transfer to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                predicted = net(inputs)
                #predicted = softmax(outputs)
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
                inputs = inputs.to(device)
                predicted = net(inputs)
                #predicted = softmax(outputs)
                sub_trainlabel[start:end] = predicted.cpu()
                sub_trainimg[start:end] = inputs.cpu()
            
    return sub_trainimg,sub_trainlabel


def label_dataset_onelabel(net, sub_trainloader, initial_label = False, nb_class = 10, device = torch.device("cuda:0")):
    softmax = nn.Softmax()
    shape = sub_trainloader.dataset[0][0].shape
    sub_trainimg = torch.zeros(len(sub_trainloader.dataset),shape[0],shape[1],shape[2])
    sub_trainlabel = torch.zeros(len(sub_trainloader.dataset)).long()
    batch_size = sub_trainloader.batch_size
    if(initial_label == True):
        for i, (inputs, labels) in enumerate(sub_trainloader, 0):
            start = batch_size * i
            end = min(batch_size * (i+1), len(sub_trainloader.dataset))
            net.eval()
            with torch.no_grad():
                # Transfer to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                met = net.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
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
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                sub_trainlabel[start:end] = predicted.cpu()
                sub_trainimg[start:end] = inputs.cpu()
            
    return sub_trainimg,sub_trainlabel.long()


def eval_training(net, test_loader, test_loss_function=nn.CrossEntropyLoss(), device = torch.device("cuda:0")):

    net.eval()
    net = net.to(device)

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = test_loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Evaluating Network.....')
    print('Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)))
    print()
    return correct.float() / len(test_loader.dataset)


def train_substitute_softlabel_onepatch(oracle_model, extraction_model, train_loader, MAX_ROUND = 5, 
                                        n_class = 10,device =  torch.device("cuda:0"), save_path = "",
                                        warm_start = 1):
    #initiate
    dataset = train_loader.dataset
    oracle_model = oracle_model.to(device)
    input_shape = list(dataset[0][0].shape)
    model = extraction_model.to(device)
    
     #Load test dataset
    test_dataloader = get_test_dataloader_cifar10(
            0,
            1,
            num_workers=8,
            batch_size=128,
            shuffle=False
        )
    test_dummy_img, test_dummy_label = label_dataset_onelabel(oracle_model, test_dataloader,True)
    test_dummy_dataset = torch.utils.data.TensorDataset(test_dummy_img,test_dummy_label)
    test_dummy_dataloader = torch.utils.data.DataLoader(test_dummy_dataset, batch_size=128, shuffle=True)
    
    #Extraction start
    for round in range(MAX_ROUND):
        print("===round:",round,"start===") 
        
        #if warm start, do not calculate AE
        if(round <= warm_start):
            dummy_img, dummy_label = label_dataset_onelabel(oracle_model, train_loader,True)
            dummy_dataset = torch.utils.data.TensorDataset(dummy_img,dummy_label)
            dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=128, shuffle=True)
            train.main(model, "extr_normal_warm" + str(round), 3, 
                                    lr = 0.01, newloader = dummy_dataloader, b = 128, 
                       save_path = save_path, normalize = True, epoch = 100, test_dataloader = test_dataloader)
            recent_folder = train.most_recent_folder(os.path.join(save_path,"extr_normal_warm" + str(round)), fmt= '%A_%d_%B_%Y_%Hh_%Mm_%Ss')
            best_weights = train.best_acc_weights(os.path.join(save_path,"extr_normal_warm" + str(round), recent_folder))
            weights_path = os.path.join(save_path, "extr_normal_warm" + str(round), recent_folder, best_weights)
            print("load model from path:",weights_path)
            model.load_state_dict(torch.load(weights_path))
        
        else:
            for batch_index, (images, labels) in enumerate(train_loader):
                
                 #Train model with normal data
                batch_dataset = torch.utils.data.TensorDataset(images, labels)
                batch_dataloader = torch.utils.data.DataLoader(batch_dataset, batch_size=128, shuffle=True)
                dummy_batch_img, dummy_batch_label = label_dataset_softlabel(oracle_model, batch_dataloader,True)
                dummy_batch_dataset = torch.utils.data.TensorDataset(dummy_batch_img,dummy_batch_label)
                dummy_batch_dataloader = torch.utils.data.DataLoader(dummy_batch_dataset, batch_size=128, shuffle=True)
                train_cifar10.main(model, "extr_normal_" + str(round) + "_" + str(batch_index), 3, 
                                        lr = 0.001, newloader = dummy_batch_dataloader, b = 128, save_path = save_path, 
                                        normalize = True, device = device, epoch = 20, test_dataloader = test_dummy_dataloader,
                                        CIFAR10_MILESTONES = [10, 10])
                recent_folder = train_cifar10.most_recent_folder(os.path.join(save_path,"extr_normal_" + str(round) + "_" + str(batch_index)), fmt= '%A_%d_%B_%Y_%Hh_%Mm_%Ss')
                best_weights = train_cifar10.best_acc_weights(os.path.join(save_path,"extr_normal_" + str(round)  + "_" + str(batch_index), recent_folder))
                weights_path = os.path.join(save_path, "extr_normal_" + str(round)  + "_" + str(batch_index), recent_folder, best_weights)
                print("load model from path:",weights_path)
                model.load_state_dict(torch.load(weights_path))
                dummy_batch_img1, dummy_batch_label1 = label_dataset_onelabel(oracle_model, batch_dataloader,True)
                dummy_batch_dataset1 = torch.utils.data.TensorDataset(dummy_batch_img1,dummy_batch_label1)
                dummy_batch_dataloader1 = torch.utils.data.DataLoader(dummy_batch_dataset1, batch_size=128, shuffle=True)
                #eval the trained model
                r1 = eval_training(model, batch_dataloader, device = device)
                r2 = eval_training(model, dummy_batch_dataloader1, device = device)
                print("acc for groundtruth on normal dataset:", r1, "acc for extraction on normal dataset:", r2)
                
                #Generate adversarial example
                print("data augmentation using cw process begins")
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                classifier = PyTorchClassifier(
                    model=model,
                    loss=criterion,
                    optimizer=optimizer,
                    input_shape=(3, 32, 32),
                    nb_classes=10,
                    device_type = device 
                )
                attack = CarliniL2Method(classifier, confidence = 0.0, batch_size = 32)
                img = []
                adv_img = []
                l = []
                for images, labels in batch_dataloader:
                    images_np = images.cpu().numpy()
                    # 然后使用转换后的 numpy 数组
                    adv = attack.generate(x=images_np)
                    #adv = attack.generate(x=images)
                    for cnt in range(len(labels)):
                        img.append(images[cnt])
                        adv_img.append(adv[cnt])
                        l.append(labels[cnt])
                    pickle.dump([adv_img,l], open( save_path + "/extr_" + str(round)+ "_" + str(batch_index) + "_CW2adv.pkl", "wb"))
                    pickle.dump([img,l],open(save_path + "/extr_" + str(round)+ "_" + str(batch_index) + "_CW2cln.pkl", "wb"))
                adv,lab = pickle.load(open(save_path + "/extr_" + str(round)+ "_" + str(batch_index) + "_CW2adv.pkl", 'rb'))
                adv = np.stack(adv)
                lab = torch.stack(l).reshape(-1)
                adv = torch.from_numpy(adv)
                

                #Train model with normal + adv in this batch

                adv_dataset = udata.TensorDataset(adv,lab)
                adv_dataloader =  torch.utils.data.DataLoader(adv_dataset, batch_size=128, shuffle=True)
                r1 = eval_training(model, adv_dataloader, device = device)
                print("The attack success rate is: ", 1-r1)
                adv_dummy_img, adv_dummy_label = label_dataset_softlabel(oracle_model, adv_dataloader,True)
                adv_dummy_dataset = torch.utils.data.TensorDataset(adv_dummy_img,adv_dummy_label)
                adv_dummy_dataloader =  torch.utils.data.DataLoader(adv_dummy_dataset, batch_size=128, shuffle=True)
                adv_dummy_img1, adv_dummy_label1 = label_dataset_onelabel(oracle_model, adv_dataloader,True)
                adv_dummy_dataset1 = torch.utils.data.TensorDataset(adv_dummy_img1,adv_dummy_label1)
                adv_dummy_dataloader1 =  torch.utils.data.DataLoader(adv_dummy_dataset1, batch_size=128, shuffle=True)
                concat_data = udata.ConcatDataset([adv_dummy_dataset, dummy_batch_dataset])                          
                print("Train model with number of data:", len(concat_data))
                concat_train_loader = torch.utils.data.DataLoader(concat_data, batch_size=128, shuffle=True)
                train_cifar10.main(model, "extr_adv_" + str(round) + "_" + str(batch_index), 3, 
                                        lr = 0.001, newloader = concat_train_loader, b = 128, save_path = save_path, 
                                        normalize = True, device = device, epoch = 50, test_dataloader = test_dummy_dataloader,
                                    CIFAR10_MILESTONES = [20, 30])
                recent_folder = train_cifar10.most_recent_folder(os.path.join(save_path,"extr_adv_" + str(round) + "_" + str(batch_index)), fmt= '%A_%d_%B_%Y_%Hh_%Mm_%Ss')
                best_weights = train_cifar10.best_acc_weights(os.path.join(save_path,"extr_adv_" + str(round) + "_" + str(batch_index) , recent_folder))
                weights_path = os.path.join(save_path,"extr_adv_" + str(round) + "_" + str(batch_index), recent_folder, best_weights)
                model.load_state_dict(torch.load(weights_path))
                print("load model from path:",weights_path)
                                
                #eval trained network
                concat_data1 = udata.ConcatDataset([adv_dummy_dataset1, dummy_batch_dataset1])
                concat_train_loader1 = torch.utils.data.DataLoader(concat_data1, batch_size=32, shuffle=True)
                r1 = eval_training(oracle_model, adv_dataloader, device = device)
                r2 = eval_training(model, adv_dummy_dataloader1, device = device)
                r3 = eval_training(model, concat_train_loader1, device = device)
                r4 = eval_training(model, dummy_batch_dataloader1, device = device)
                print("acc of adv on oracle(tranferability):", 1-r1, "acc for extraction on adv dataset:", r2, "acc for extraction on concat dataset:", r3,"acc for extraction on normal dataset:", r4)


    return model


import random
import math 
class model_dict:
    def __init__(self):
        self.p = random.uniform(0.1,0.15)
        self.batch_size = random.randint(1,4) * 64
        optimizer = ["SGD","Adam","RMSprop"]
        self.optimizer = random.choice(optimizer) 
    def itera(self,p):
        if (math.ceil(0.4/p) >= math.floor(0.5/p)):
            print(p) 
            MAX_ITER = math.ceil(0.4/(p))
        else: 
            print(p, math.ceil(0.4/(p)), math.floor((0.5/p)))
            MAX_ITER = random.randint(math.ceil(0.4/(p)),math.floor((0.5/p)))
        return MAX_ITER 


print('___start____')
torch.cuda.set_device(1)
device = "cuda:1"
data_dir = os.path.join("./data", 'CIFAR10')
transform_test = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
        
    ])
fv = get_model("resnet18",True, 10)
subset1_mean,subset2_mean = [0.49116027,0.49106753]
subset1_std, subset2_std = [0.24728487,0.24676652]
fv.load_state_dict(torch.load("/home/fazhong/studio/uap/checkpoint/fv_resnet18/Tuesday_19_March_2024_13h_44m_32s/training_session_1-98-best.pth"))
fv = nn.Sequential(transforms.Normalize(subset1_mean, subset1_std), fv)
cifar10_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_test)

for cnt in range(10):
        print("=====================model extraction " + str(cnt) + "==============================")
        extraction_model = get_model("resnet34",True,10)
        extraction_model.load_state_dict(torch.load("/home/fazhong/studio/uap/checkpoint/fv_resnet34/Wednesday_20_March_2024_01h_06m_45s/fv_resnet34-94-best.pth"))
        extraction_model = nn.Sequential(transforms.Normalize(subset1_mean, subset1_std),extraction_model)
        
        
        save_path =  "./checkpoint/softlabel_resnet34" 
        config = model_dict() 
        file_handler = open(save_path + "/log_" + str(cnt) + ".pkl", "wb")
        pickle.dump(config, file_handler)
    
        train_size = int(config.p*len(cifar10_trainset)) 
        sub_train_size = len(cifar10_trainset) - train_size 
        train_dataset, sub_train_dataset = torch.utils.data.random_split(cifar10_trainset, [train_size, sub_train_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,shuffle = False)
        
        extraction_model = train_substitute_softlabel_onepatch(fv, extraction_model, 
                            train_loader,MAX_ROUND=config.itera(config.p),device = torch.device("cuda:1"), 
                           save_path = "./checkpoint/softlabel_resnet34_cuda1/extmodel_" + str(cnt) + "/",
                           warm_start = -1)      


# fv = get_model("resnet34",True,10)
# fv


