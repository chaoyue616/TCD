#!/usr/bin/python
#coding:utf8
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import torch 
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix
import math
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import random
import vit
from vit import AdversarialNetwork
from vit import FC
from vit import AdversarialNetwork_consistency
import lossZoo

import utils
import scipy.io as scio
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#parameter setting
lr = 0.001
batch_size = 50
num_epochs = 160




# --------------------------- train=test=setting ------------------------  #

temfile_pathp = 'G:\\paperdata\\MultipleSensors'

# --------------------------- sensor 1 ------------------------  #
#1800_train data
data_train_1800_1 = sio.loadmat(str(temfile_pathp) + '\\1800train_data_3.mat')
train_data_1800_1 = data_train_1800_1['data']
train_label_1800_1 = data_train_1800_1['label']
num_train_instances_1800_1 = len(train_data_1800_1)
train_data_1800_1 = torch.from_numpy(train_data_1800_1).type(torch.FloatTensor)
train_label_1800_1 = torch.from_numpy(train_label_1800_1).type(torch.LongTensor)
train_data_1800_1 = train_data_1800_1.view(num_train_instances_1800_1, 1, -1)
train_label_1800_1 = torch.topk(train_label_1800_1, 1)[1].squeeze(1)
train_label_1800_1 = train_label_1800_1.view(num_train_instances_1800_1)
train_dataset_1800_1 = TensorDataset(train_data_1800_1, train_label_1800_1)
train_data_loader_1800_1 = DataLoader(dataset=train_dataset_1800_1, batch_size=batch_size,shuffle=True)

#1800_test data
data_test_1800_1 = sio.loadmat(str(temfile_pathp) + '\\1800test_data_3.mat')
test_data_1800_1 = data_test_1800_1['data']
test_label_1800_1 = data_test_1800_1['label']
num_test_instances_1800_1 = len(test_data_1800_1)
test_data_1800_1 = torch.from_numpy(test_data_1800_1).type(torch.FloatTensor)
test_label_1800_1 = torch.from_numpy(test_label_1800_1).type(torch.LongTensor)
test_data_1800_1 = test_data_1800_1.view(num_test_instances_1800_1, 1, -1)
test_label_1800_1 = torch.topk(test_label_1800_1, 1)[1].squeeze(1)
test_label_1800_1 = test_label_1800_1.view(num_test_instances_1800_1)
test_dataset_1800_1 = TensorDataset(test_data_1800_1, test_label_1800_1)
test_data_loader_1800_1 = DataLoader(dataset=test_dataset_1800_1, batch_size=batch_size,shuffle=True)



#2400_train data
data_train_2400_1 = sio.loadmat(str(temfile_pathp) + '\\2400train_data_3.mat')
train_data_2400_1 = data_train_2400_1['data']
train_label_2400_1 = data_train_2400_1['label']
num_train_instances_2400_1 = len(train_data_2400_1)
train_data_2400_1 = torch.from_numpy(train_data_2400_1).type(torch.FloatTensor)
train_label_2400_1 = torch.from_numpy(train_label_2400_1).type(torch.LongTensor)
train_data_2400_1 = train_data_2400_1.view(num_train_instances_2400_1, 1, -1)
train_label_2400_1 = torch.topk(train_label_2400_1, 1)[1].squeeze(1)
train_label_2400_1 = train_label_2400_1.view(num_train_instances_2400_1)
train_dataset_2400_1 = TensorDataset(train_data_2400_1, train_label_2400_1)
train_data_loader_2400_1 = DataLoader(dataset=train_dataset_2400_1, batch_size=batch_size,shuffle=True)

#2400_test data
data_test_2400_1 = sio.loadmat(str(temfile_pathp) + '\\2400test_data_3.mat')
test_data_2400_1 = data_test_2400_1['data']
test_label_2400_1 = data_test_2400_1['label']
num_test_instances_2400_1 = len(test_data_2400_1)
test_data_2400_1 = torch.from_numpy(test_data_2400_1).type(torch.FloatTensor)
test_label_2400_1 = torch.from_numpy(test_label_2400_1).type(torch.LongTensor)
test_data_2400_1 = test_data_2400_1.view(num_test_instances_2400_1, 1, -1)
test_label_2400_1 = torch.topk(test_label_2400_1, 1)[1].squeeze(1)
test_label_2400_1 = test_label_2400_1.view(num_test_instances_2400_1)
test_dataset_2400_1 = TensorDataset(test_data_2400_1, test_label_2400_1)
test_data_loader_2400_1 = DataLoader(dataset=test_dataset_2400_1, batch_size=batch_size,shuffle=True)








# --------------------------- sensor 2 ------------------------  #
#1800_train data
data_train_1800_2 = sio.loadmat(str(temfile_pathp) + '\\1800train_data_6.mat')
train_data_1800_2 = data_train_1800_2['data']
train_label_1800_2 = data_train_1800_2['label']
num_train_instances_1800_2 = len(train_data_1800_2)
train_data_1800_2 = torch.from_numpy(train_data_1800_2).type(torch.FloatTensor)
train_label_1800_2 = torch.from_numpy(train_label_1800_2).type(torch.LongTensor)
train_data_1800_2 = train_data_1800_2.view(num_train_instances_1800_2, 1, -1)
train_label_1800_2 = torch.topk(train_label_1800_2, 1)[1].squeeze(1)
train_label_1800_2 = train_label_1800_2.view(num_train_instances_1800_2)
train_dataset_1800_2 = TensorDataset(train_data_1800_2, train_label_1800_2)
train_data_loader_1800_2 = DataLoader(dataset=train_dataset_1800_2, batch_size=batch_size,shuffle=True)

#1800_test data
data_test_1800_2 = sio.loadmat(str(temfile_pathp) + '\\1800test_data_6.mat')
test_data_1800_2 = data_test_1800_2['data']
test_label_1800_2 = data_test_1800_2['label']
num_test_instances_1800_2 = len(test_data_1800_2)
test_data_1800_2 = torch.from_numpy(test_data_1800_2).type(torch.FloatTensor)
test_label_1800_2 = torch.from_numpy(test_label_1800_2).type(torch.LongTensor)
test_data_1800_2 = test_data_1800_2.view(num_test_instances_1800_2, 1, -1)
test_label_1800_2 = torch.topk(test_label_1800_2, 1)[1].squeeze(1)
test_label_1800_2 = test_label_1800_2.view(num_test_instances_1800_2)
test_dataset_1800_2 = TensorDataset(test_data_1800_2, test_label_1800_2)
test_data_loader_1800_2 = DataLoader(dataset=test_dataset_1800_2, batch_size=batch_size,shuffle=True)



#2400_train data
data_train_2400_2 = sio.loadmat(str(temfile_pathp) + '\\2400train_data_6.mat')
train_data_2400_2 = data_train_2400_2['data']
train_label_2400_2 = data_train_2400_2['label']
num_train_instances_2400_2 = len(train_data_2400_2)
train_data_2400_2 = torch.from_numpy(train_data_2400_2).type(torch.FloatTensor)
train_label_2400_2 = torch.from_numpy(train_label_2400_2).type(torch.LongTensor)
train_data_2400_2 = train_data_2400_2.view(num_train_instances_2400_2, 1, -1)
train_label_2400_2 = torch.topk(train_label_2400_2, 1)[1].squeeze(1)
train_label_2400_2 = train_label_2400_2.view(num_train_instances_2400_2)
train_dataset_2400_2 = TensorDataset(train_data_2400_2, train_label_2400_2)
train_data_loader_2400_2 = DataLoader(dataset=train_dataset_2400_2, batch_size=batch_size,shuffle=True)

#2400_test data
data_test_2400_2 = sio.loadmat(str(temfile_pathp) + '\\2400test_data_6.mat')
test_data_2400_2 = data_test_2400_2['data']
test_label_2400_2 = data_test_2400_2['label']
num_test_instances_2400_2 = len(test_data_2400_2)
test_data_2400_2 = torch.from_numpy(test_data_2400_2).type(torch.FloatTensor)
test_label_2400_2 = torch.from_numpy(test_label_2400_2).type(torch.LongTensor)
test_data_2400_2 = test_data_2400_2.view(num_test_instances_2400_2, 1, -1)
test_label_2400_2 = torch.topk(test_label_2400_2, 1)[1].squeeze(1)
test_label_2400_2 = test_label_2400_2.view(num_test_instances_2400_2)
test_dataset_2400_2 = TensorDataset(test_data_2400_2, test_label_2400_2)
test_data_loader_2400_2 = DataLoader(dataset=test_dataset_2400_2, batch_size=batch_size,shuffle=True)





# --------------------------- sensor 3 ------------------------  #
#1800_train data
data_train_1800_3 = sio.loadmat(str(temfile_pathp) + '\\1800train_data_3.mat')
train_data_1800_3 = data_train_1800_3['data']
train_label_1800_3 = data_train_1800_3['label']
num_train_instances_1800_3 = len(train_data_1800_3)
train_data_1800_3 = torch.from_numpy(train_data_1800_3).type(torch.FloatTensor)
train_label_1800_3 = torch.from_numpy(train_label_1800_3).type(torch.LongTensor)
train_data_1800_3 = train_data_1800_3.view(num_train_instances_1800_3, 1, -1)
train_label_1800_3 = torch.topk(train_label_1800_3, 1)[1].squeeze(1)
train_label_1800_3 = train_label_1800_3.view(num_train_instances_1800_3)
train_dataset_1800_3 = TensorDataset(train_data_1800_3, train_label_1800_3)
train_data_loader_1800_3 = DataLoader(dataset=train_dataset_1800_3, batch_size=batch_size,shuffle=True)

#1800_test data
data_test_1800_3 = sio.loadmat(str(temfile_pathp) + '\\1800test_data_3.mat')
test_data_1800_3 = data_test_1800_3['data']
test_label_1800_3 = data_test_1800_3['label']
num_test_instances_1800_3 = len(test_data_1800_3)
test_data_1800_3 = torch.from_numpy(test_data_1800_3).type(torch.FloatTensor)
test_label_1800_3 = torch.from_numpy(test_label_1800_3).type(torch.LongTensor)
test_data_1800_3 = test_data_1800_3.view(num_test_instances_1800_3, 1, -1)
test_label_1800_3 = torch.topk(test_label_1800_3, 1)[1].squeeze(1)
test_label_1800_3 = test_label_1800_3.view(num_test_instances_1800_3)
test_dataset_1800_3 = TensorDataset(test_data_1800_3, test_label_1800_3)
test_data_loader_1800_3 = DataLoader(dataset=test_dataset_1800_3, batch_size=batch_size,shuffle=True)



#2400_train data
data_train_2400_3 = sio.loadmat(str(temfile_pathp) + '\\2400train_data_3.mat')
train_data_2400_3 = data_train_2400_3['data']
train_label_2400_3 = data_train_2400_3['label']
num_train_instances_2400_3 = len(train_data_2400_3)
train_data_2400_3 = torch.from_numpy(train_data_2400_3).type(torch.FloatTensor)
train_label_2400_3 = torch.from_numpy(train_label_2400_3).type(torch.LongTensor)
train_data_2400_3 = train_data_2400_3.view(num_train_instances_2400_3, 1, -1)
train_label_2400_3 = torch.topk(train_label_2400_3, 1)[1].squeeze(1)
train_label_2400_3 = train_label_2400_3.view(num_train_instances_2400_3)
train_dataset_2400_3 = TensorDataset(train_data_2400_3, train_label_2400_3)
train_data_loader_2400_3 = DataLoader(dataset=train_dataset_2400_3, batch_size=batch_size,shuffle=True)

#2400_test data
data_test_2400_3 = sio.loadmat(str(temfile_pathp) + '\\2400test_data_3.mat')
test_data_2400_3 = data_test_2400_3['data']
test_label_2400_3 = data_test_2400_3['label']
num_test_instances_2400_3 = len(test_data_2400_3)
test_data_2400_3 = torch.from_numpy(test_data_2400_3).type(torch.FloatTensor)
test_label_2400_3 = torch.from_numpy(test_label_2400_3).type(torch.LongTensor)
test_data_2400_3 = test_data_2400_3.view(num_test_instances_2400_3, 1, -1)
test_label_2400_3 = torch.topk(test_label_2400_3, 1)[1].squeeze(1)
test_label_2400_3 = test_label_2400_3.view(num_test_instances_2400_3)
test_dataset_2400_3 = TensorDataset(test_data_2400_3, test_label_2400_3)
test_data_loader_2400_3 = DataLoader(dataset=test_dataset_2400_3, batch_size=batch_size,shuffle=True)





hidden_size=128

ad_net_patch = AdversarialNetwork(hidden_size//8, hidden_size=10)
ad_net_patch = ad_net_patch.to(DEVICE)


ad_net_class = AdversarialNetwork(hidden_size, hidden_size=10)
ad_net_class = ad_net_class.to(DEVICE)

ad_consistency = AdversarialNetwork_consistency(16, hidden_size_co=10)
ad_consistency = ad_consistency.to(DEVICE)



Transformer_net_1 = ViT(num_patches = 16, dim = 128)
Transformer_net_2 = ViT(num_patches = 16, dim = 128)
Transformer_net_3 = ViT(num_patches = 16, dim = 128)
Transformer_net_1 = Transformer_net_1.to(DEVICE)
Transformer_net_2 = Transformer_net_2.to(DEVICE)
Transformer_net_3 = Transformer_net_3.to(DEVICE)


# 训练CNN

FC_1 = FC()
FC_2 = FC()
FC_3 = FC()
FC_1 = FC_1.to(DEVICE)
FC_2 = FC_2.to(DEVICE)
FC_3 = FC_3.to(DEVICE)




criterion = torch.nn.CrossEntropyLoss()




for epoch in range(num_epochs):

    LEARNING_RATE = lr * math.pow(0.5, math.floor(epoch / 40))
    optimizer = torch.optim.Adam(list(Transformer_net_1.parameters())+list(Transformer_net_2.parameters())+list(Transformer_net_3.parameters())
                                 + list(FC_1.parameters()) + list(FC_2.parameters()) + list(FC_3.parameters()) + list(ad_net_patch.parameters()) + list(ad_net_class.parameters())
                                 + list(ad_consistency.parameters())
                                 , lr=LEARNING_RATE)

    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    loss_ad_class_all_1 = 0
    loss_ad_class_all_2 = 0
    loss_ad_class_all_3 = 0
    correct_tar = 0
    total_loss_train = 0


    source_iter_1 = iter(train_data_loader_1800_1)
    target_iter_1 = iter(train_data_loader_2400_1)

    source_iter_2 = iter(train_data_loader_1800_2)
    target_iter_2 = iter(train_data_loader_2400_2)

    source_iter_3 = iter(train_data_loader_1800_3)
    target_iter_3 = iter(train_data_loader_2400_3)


    Transformer_net_1.train()
    Transformer_net_2.train()
    Transformer_net_3.train()
    FC_1.train()
    FC_2.train()
    FC_3.train()
    ad_net_patch.train()
    ad_net_class.train()
    ad_consistency.train()



    # train
    for idx in range(len(train_data_loader_1800_1)):
        # print(idx)
        sdata_1 = next(source_iter_1)
        tdata_1 = next(target_iter_1)
        input1_1, label1_1 = sdata_1
        input2_1, label2_1 = tdata_1
        input1_1 = Variable(input1_1.to(DEVICE))
        label1_1 = Variable(label1_1.to(DEVICE))
        input2_1 = Variable(input2_1.to(DEVICE))
        label2_1 = Variable(label2_1.to(DEVICE))



        sdata_2 = next(source_iter_2)
        tdata_2 = next(target_iter_2)
        input1_2, label1_2 = sdata_2
        input2_2, label2_2 = tdata_2
        input1_2 = Variable(input1_2.to(DEVICE))
        label1_2 = Variable(label1_2.to(DEVICE))
        input2_2 = Variable(input2_2.to(DEVICE))
        label2_2 = Variable(label2_2.to(DEVICE))


        sdata_3 = next(source_iter_3)
        tdata_3 = next(target_iter_3)
        input1_3, label1_3 = sdata_3
        input2_3, label2_3 = tdata_3
        input1_3 = Variable(input1_3.to(DEVICE))
        label1_3 = Variable(label1_3.to(DEVICE))
        input2_3 = Variable(input2_3.to(DEVICE))
        label2_3 = Variable(label2_3.to(DEVICE))




        optimizer.zero_grad()


        src_feature_1,tar_feature_1, loss_ad_1 = Transformer_net_1(input1_1,input2_1,ad_net_patch)
        src_feature_2, tar_feature_2, loss_ad_2 = Transformer_net_2(input1_2, input2_2, ad_net_patch)
        src_feature_3, tar_feature_3, loss_ad_3 = Transformer_net_3(input1_3, input2_3, ad_net_patch)


        pred_src_1 = FC_1(src_feature_1)
        pred_src_2 = FC_2(src_feature_2)
        pred_src_3 = FC_3(src_feature_3)


        pred_tar_1 = FC_1(tar_feature_1)
        pred_tar_2 = FC_2(tar_feature_2)
        pred_tar_3 = FC_3(tar_feature_3)


        class_loss_1 = criterion(pred_src_1, label1_1)
        class_loss_2 = criterion(pred_src_2, label1_2)
        class_loss_3 = criterion(pred_src_3, label1_3)


        tar_feature = torch.cat((pred_tar_1, pred_tar_2, pred_tar_3), 0)



        loss_ad_consistency = lossZoo.adv_co(tar_feature, ad_consistency)


        loss_ad_class_1 = lossZoo.adv(torch.cat((src_feature_1, tar_feature_1), 0), ad_net_class)
        loss_ad_class_2 = lossZoo.adv(torch.cat((src_feature_2, tar_feature_1), 0), ad_net_class)
        loss_ad_class_3 = lossZoo.adv(torch.cat((src_feature_3, tar_feature_1), 0), ad_net_class)

        loss_all_class = class_loss_1 + class_loss_2 + class_loss_3
        loss_ad_patch = loss_ad_1 + loss_ad_2 + loss_ad_3
        loss_ad_class = loss_ad_class_1 + loss_ad_class_2 + loss_ad_class_3


        loss = loss_all_class + 1 * loss_ad_patch + 1 * loss_ad_class + 1 * loss_ad_consistency

        # print(loss)

        loss.backward()
        optimizer.step()



        pred_s_1 = pred_src_1.data.max(1)[1]
        correct_1 += pred_s_1.eq(label1_1.data.view_as(pred_s_1)).cpu().sum()
        pred_s_2 = pred_src_2.data.max(1)[1]
        correct_2 += pred_s_2.eq(label1_2.data.view_as(pred_s_2)).cpu().sum()
        pred_s_3 = pred_src_3.data.max(1)[1]
        correct_3 += pred_s_3.eq(label1_3.data.view_as(pred_s_3)).cpu().sum()
        total_loss_train += loss.data \

        loss_ad_class_all_1 += loss_ad_class_1
        loss_ad_class_all_2 += loss_ad_class_2
        loss_ad_class_all_3 += loss_ad_class_3






    total_loss_train /= len(train_data_loader_1800_1)
    acc_1 = 100 * float(correct_1) / len(train_data_loader_1800_1.dataset)  # 源域准确率
    acc_2 = 100 * float(correct_2) / len(train_data_loader_1800_2.dataset)  # 源域准确率
    acc_3 = 100 * float(correct_3) / len(train_data_loader_1800_3.dataset)  # 源域准确率

    res_e = '==train==source domain==Epoch: [{}/{}], training loss: {:.4f}, correct_1: {:.2f}%, correct_2: {:.2f}%, correct_3: {:.2f}%, test accuracy: {:.2f}%'.format(
        epoch, num_epochs, total_loss_train, acc_1, acc_2, acc_3, acc_3,)
    tqdm.write(res_e)


    #test
    Transformer_net_1.eval()
    Transformer_net_2.eval()
    Transformer_net_3.eval()
    FC_1.eval()
    FC_2.eval()
    FC_3.eval()
    ad_net_patch.eval()
    ad_net_class.eval()
    ad_consistency.eval()



    source_iter_1_t = iter(test_data_loader_1800_1)
    target_iter_1_t = iter(test_data_loader_2400_1)

    source_iter_2_t = iter(test_data_loader_1800_2)
    target_iter_2_t = iter(test_data_loader_2400_2)

    source_iter_3_t = iter(test_data_loader_1800_3)
    target_iter_3_t = iter(test_data_loader_2400_3)

    correct_test = 0
    total_loss_test = 0
    correct_1_t = 0
    correct_2_t = 0
    correct_3_t = 0
    acc_test = 0



    with torch.no_grad():
        for idx in range(len(test_data_loader_2400_1)):


            sdata_1_t = next(source_iter_1_t)
            tdata_1_t = next(target_iter_1_t)
            input1_1_t, label1_1_t = sdata_1_t
            input2_1_t, label2_1_t = tdata_1_t
            input1_1_t = Variable(input1_1_t.to(DEVICE))
            label1_1_t = Variable(label1_1_t.to(DEVICE))
            input2_1_t = Variable(input2_1_t.to(DEVICE))
            label2_1_t = Variable(label2_1_t.to(DEVICE))

            sdata_2_t = next(source_iter_2_t)
            tdata_2_t = next(target_iter_2_t)
            input1_2_t, label1_2_t = sdata_2_t
            input2_2_t, label2_2_t = tdata_2_t
            input1_2_t = Variable(input1_2_t.to(DEVICE))
            label1_2_t = Variable(label1_2_t.to(DEVICE))
            input2_2_t = Variable(input2_2_t.to(DEVICE))
            label2_2_t = Variable(label2_2_t.to(DEVICE))

            sdata_3_t = next(source_iter_3_t)
            tdata_3_t = next(target_iter_3_t)
            input1_3_t, label1_3_t = sdata_3_t
            input2_3_t, label2_3_t = tdata_3_t
            input1_3_t = Variable(input1_3_t.to(DEVICE))
            label1_3_t = Variable(label1_3_t.to(DEVICE))
            input2_3_t = Variable(input2_3_t.to(DEVICE))
            label2_3_t = Variable(label2_3_t.to(DEVICE))



            src_feature_1_t, tar_feature_1_t, _ = Transformer_net_1(input1_1_t, input2_1_t, ad_net_patch)
            src_feature_2_t, tar_feature_2_t, _ = Transformer_net_2(input1_2_t, input2_2_t, ad_net_patch)
            src_feature_3_t, tar_feature_3_t, _ = Transformer_net_3(input1_3_t, input2_3_t, ad_net_patch)

            pred_src_1_t = FC_1(src_feature_1_t)
            pred_src_2_t = FC_2(src_feature_2_t)
            pred_src_3_t = FC_3(src_feature_3_t)

            pred_tar_1_t = FC_1(tar_feature_1_t)
            pred_tar_2_t = FC_2(tar_feature_2_t)
            pred_tar_3_t = FC_3(tar_feature_3_t)


            pred_s_1_t = pred_src_1_t.data.max(1)[1]
            correct_1_t += pred_s_1_t.eq(label1_1_t.data.view_as(pred_s_1_t)).cpu().sum()
            pred_s_2_t = pred_src_2_t.data.max(1)[1]
            correct_2_t += pred_s_2_t.eq(label1_2_t.data.view_as(pred_s_2_t)).cpu().sum()
            pred_s_3_t = pred_src_3_t.data.max(1)[1]
            correct_3_t += pred_s_3_t.eq(label1_3_t.data.view_as(pred_s_3_t)).cpu().sum()


        acc_1_t = 100 * float(correct_1_t) / len(test_data_loader_1800_1.dataset)  # 源域准确率
        acc_2_t = 100 * float(correct_2_t) / len(test_data_loader_1800_2.dataset)  # 源域准确率
        acc_3_t = 100 * float(correct_3_t) / len(test_data_loader_1800_3.dataset)  # 源域准确率


        w_1 = (acc_1_t * (1 / loss_ad_class_all_1)) / (acc_1_t * (1 / loss_ad_class_all_1) + acc_2_t * (1 / loss_ad_class_all_2) + acc_3_t * (1 / loss_ad_class_all_3))
        w_2 = (acc_2_t * (1 / loss_ad_class_all_2)) / (acc_1_t * (1 / loss_ad_class_all_1) + acc_2_t * (1 / loss_ad_class_all_2) + acc_3_t * (1 / loss_ad_class_all_3))
        w_3 = (acc_3_t * (1 / loss_ad_class_all_3)) / (acc_1_t * (1 / loss_ad_class_all_1) + acc_2_t * (1 / loss_ad_class_all_2) + acc_3_t * (1 / loss_ad_class_all_3))

        res_e = '==test==source domain==Epoch: [{}/{}], correct_1: {:.2f}%, correct_2: {:.2f}%, correct_3: {:.2f}%'.format(
            epoch, num_epochs, acc_1_t, acc_2_t, acc_3_t)
        tqdm.write(res_e)




    source_iter_1_t = iter(test_data_loader_1800_1)
    target_iter_1_t = iter(test_data_loader_2400_1)

    source_iter_2_t = iter(test_data_loader_1800_2)
    target_iter_2_t = iter(test_data_loader_2400_2)

    source_iter_3_t = iter(test_data_loader_1800_3)
    target_iter_3_t = iter(test_data_loader_2400_3)

    with torch.no_grad():
        for idx in range(len(test_data_loader_2400_1)):
            sdata_1_t = next(source_iter_1_t)
            tdata_1_t = next(target_iter_1_t)
            input1_1_t, label1_1_t = sdata_1_t
            input2_1_t, label2_1_t = tdata_1_t
            input1_1_t = Variable(input1_1_t.to(DEVICE))
            label1_1_t = Variable(label1_1_t.to(DEVICE))
            input2_1_t = Variable(input2_1_t.to(DEVICE))
            label2_1_t = Variable(label2_1_t.to(DEVICE))

            sdata_2_t = next(source_iter_2_t)
            tdata_2_t = next(target_iter_2_t)
            input1_2_t, label1_2_t = sdata_2_t
            input2_2_t, label2_2_t = tdata_2_t
            input1_2_t = Variable(input1_2_t.to(DEVICE))
            label1_2_t = Variable(label1_2_t.to(DEVICE))
            input2_2_t = Variable(input2_2_t.to(DEVICE))
            label2_2_t = Variable(label2_2_t.to(DEVICE))

            sdata_3_t = next(source_iter_3_t)
            tdata_3_t = next(target_iter_3_t)
            input1_3_t, label1_3_t = sdata_3_t
            input2_3_t, label2_3_t = tdata_3_t
            input1_3_t = Variable(input1_3_t.to(DEVICE))
            label1_3_t = Variable(label1_3_t.to(DEVICE))
            input2_3_t = Variable(input2_3_t.to(DEVICE))
            label2_3_t = Variable(label2_3_t.to(DEVICE))

            src_feature_1_t, tar_feature_1_t, _ = Transformer_net_1(input1_1_t, input2_1_t, ad_net_patch)
            src_feature_2_t, tar_feature_2_t, _ = Transformer_net_2(input1_2_t, input2_2_t, ad_net_patch)
            src_feature_3_t, tar_feature_3_t, _ = Transformer_net_3(input1_3_t, input2_3_t, ad_net_patch)

            pred_tar_1_t = FC_1(tar_feature_1_t)
            pred_tar_2_t = FC_2(tar_feature_2_t)
            pred_tar_3_t = FC_3(tar_feature_3_t)

            pred_tar_test = w_1 * pred_tar_1_t + w_2 * pred_tar_2_t + w_3 * pred_tar_3_t

            pred_s_1_test = pred_tar_test.data.max(1)[1]
            correct_test += pred_s_1_test.eq(label2_1_t.data.view_as(pred_s_1_test)).cpu().sum()



        acc_test = 100 * float(correct_test) / len(test_data_loader_2400_1.dataset)


        res_e = '==test==target domain==Epoch: [{}/{}], test accuracy: {:.2f}%'.format(
            epoch, num_epochs, acc_test)
        tqdm.write(res_e)



