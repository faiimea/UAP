
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
import os
from uap.train.model_structure import get_model
from encoder import classifier
from encoder_training import train, evaluate
from utils import *
from encoder_trainingdata_preparation import concanate_fingerprint
import pickle
import random

# 加载您之前保存的 fingerprint points
with open("/home/fazhong/studio/uap/result/split_sets_150points_200views.pkl", "rb") as file:
    fingerprint_points = pickle.load(file)

# 现在 fingerprint_points 可用于您的 framework_building 和 framework_testing 函数
# 假设所有模型和UAP向量已经准备好


fv = get_model("resnet18",True, 10)
fv.load_state_dict(torch.load("/home/fazhong/studio/uap/checkpoint/fv_resnet18/Tuesday_19_March_2024_13h_44m_32s/training_session_1-98-best.pth"))

v_opt_np = np.load('/home/fazhong/studio/uap/v_opt.npy')
v_opt_tensor = torch.from_numpy(v_opt_np).float()
uap_vector = v_opt_tensor.to('cuda:0')  

def compare_datasets(dataset1, dataset2):
    # Compare the sizes of the two datasets
    len1 = len(dataset1)
    len2 = len(dataset2)
    print(f"Length of dataset1: {len1}")
    print(f"Length of dataset2: {len2}")
    
    # If the sizes are different, it's a clear indication that the datasets are not the same
    if len1 != len2:
        print("Datasets differ in length.")
    else:
        # Iterate over each data point if datasets have the same length
        for i in range(len1):
            features1, label1 = dataset1[i]
            features2, label2 = dataset2[i]
            print(features1,features2)
            print(label1,label2)
            # Compare the feature vectors
            
# Assuming dataset and data_set_cur are both TensorDataset objects

# 定义顶层目录
top_dir = '/home/fazhong/studio/uap/checkpoint/ext3'

# 用于存储包含"best"的文件的完整路径
best_files = []

# 遍历目录结构
for dirpath, dirnames, filenames in os.walk(top_dir):
    # 过滤出包含"best"的文件
    for filename in filenames:
        # if '100' in filename:
            # 将文件的完整路径添加到列表中
        best_files.append(os.path.join(dirpath, filename))

# 输出所有匹配的文件路径
# for file_path in best_files:
#     print(file_path)

random.shuffle(best_files)


# 随机打乱模型路径列表


# 分割模型路径为训练和测试集
half_indep = len(best_files) // 2


# 独立模型路径分割
indep_train_paths = best_files[:half_indep]
indep_test_paths = best_files[half_indep:]



# 盗版模型 - resnet34
# softlabel_resnet34_withoutadv
            
indep_model_paths = best_files[0:8]

indep_model_architectures_train = []
indep_model_architectures_test = []
for i in range(len(indep_train_paths)):
    indep_model_architectures_train.append('resnet18')
for i in range(len(indep_test_paths)):
    indep_model_architectures_test.append('resnet18')
# 同源模型 - resnet18
# resnet18_indep_n



homo_model_paths = []
top_dir = '/home/fazhong/studio/uap/checkpoint/model_homo'
for dirpath, dirnames, filenames in os.walk(top_dir):
    # 过滤出包含"best"的文件
    for filename in filenames:
        # 将文件的完整路径添加到列表中
        homo_model_paths.append(os.path.join(dirpath, filename))

random.shuffle(homo_model_paths)
half_homo = len(homo_model_paths) // 2

# 同源模型路径分割
homo_train_paths = homo_model_paths[:half_homo]
homo_test_paths = homo_model_paths[half_homo:]



homo_model_architectures_train = []
homo_model_architectures_test = []
for i in range(len(homo_train_paths)):
    homo_model_architectures_train.append('resnet18')

for i in range(len(homo_test_paths)):
    homo_model_architectures_test.append('resnet18')
    
def framework_testing(fv,
                       v, fp_gene_points, indep_models_path, indep_models_archi, homo_models_path, homo_models_archi, 
                      net, batch_size, device = torch.device('cuda:0')):
    """
    @Param:
        fv: victim model
        v:UAP vector
        fp_gene_points: result of fingerprint_point_selection
        indep_models_path: list of indep models used for testing encoder
        indep_models_archi: architecture of indep models
        homo_models_path: list of homologous models used for testing encoder
        homo_models_archi:architecture of homologous models
        net: trained encoder
        batch_size: bz of testing encoder
        device: CPU or GPU
    @Return:
        None
    """
    print('==========================================')
    print('Test Begin')
    print('==========================================')
    
    data_set = concanate_fingerprint(fv, fp_gene_points, v, 2).dataset
    fp_v = data_set[0]
    for cnt in range(len(indep_models_path)):
        f_indep = get_model(indep_models_archi[cnt],True)
        f_indep.load_state_dict(torch.load(indep_models_path[cnt]),strict=False)
        data_set_cur = concanate_fingerprint(f_indep, fp_gene_points,v, 2).dataset
        data_set = torch.utils.data.ConcatDataset([data_set,data_set_cur])
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
        # evaluate(net, fp_v, test_loader, device)
    for cnt in range(len(homo_models_path)):
        f_homo = get_model(homo_models_archi[cnt],True)
        f_homo.load_state_dict(torch.load(homo_models_path[cnt]),strict=False)
        data_set_cur = concanate_fingerprint(f_homo, fp_gene_points,v, 1).dataset
        data_set = torch.utils.data.ConcatDataset([data_set,data_set_cur])
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
        evaluate(net, fp_v, test_loader, device)



if __name__ == '__main__':
    # 构建encoder
    trained_encoder = framework_building(
        fv, 
        uap_vector, 
        fingerprint_points, 
        indep_train_paths, 
        indep_model_architectures_train, 
        homo_train_paths, 
        homo_model_architectures_train
    )

    # device = 'cuda:0'
    # trained_encoder = classifier(input_channel=3000).to(device)
    # trained_encoder.load_state_dict(torch.load('/home/fazhong/studio/uap/result/model.pth'))
    # trained_encoder = trained_encoder.to(device)
    # 测试encoder
    framework_testing(
        fv, 
        uap_vector, 
        fingerprint_points, 
        indep_test_paths, 
        indep_model_architectures_test, 
        homo_test_paths, 
        homo_model_architectures_test, 
        trained_encoder, 
        batch_size=512
    )