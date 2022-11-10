#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
from torch.nn import Softmax
import torch.nn as nn
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-10
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 


def im(outputs_test, gent=True):
    epsilon = 1e-10
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    entropy_loss = torch.mean(entropy(softmax_out))
    if gent:
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log2(msoftmax + epsilon))
        entropy_loss -= gentropy_loss
    im_loss = entropy_loss * 1.0
    return im_loss


def adv(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(features.device)
    return torch.nn.BCELoss()(ad_out, dc_target)

batch_size=50
def adv_co(features, ad_net):
    ad_out = ad_net(features)
    label_tar_1 = np.full(shape=batch_size, fill_value=0, dtype=np.int)
    label_tar_1 = torch.from_numpy(label_tar_1).type(torch.LongTensor)
    label_tar_2 = np.full(shape=batch_size, fill_value=1, dtype=np.int)
    label_tar_2 = torch.from_numpy(label_tar_2).type(torch.LongTensor)
    label_tar_3 = np.full(shape=batch_size, fill_value=2, dtype=np.int)
    label_tar_3 = torch.from_numpy(label_tar_3).type(torch.LongTensor)
    label_tar = torch.cat((label_tar_1, label_tar_2, label_tar_3), 0)
    label_tar = label_tar.to(DEVICE)
    return torch.nn.CrossEntropyLoss()(ad_out, label_tar)


def adv_local(features, ad_net, is_source=False, weights=None):
    # print("features:",features.shape)
    ad_out = ad_net(features).squeeze(3)
    batch_size = ad_out.size(0)
    num_heads = ad_out.size(1)
    seq_len = ad_out.size(2)
    
    if is_source:
        label = torch.from_numpy(np.array([[[1]*seq_len]*num_heads] * batch_size)).float().to(features.device)
    else:
        label = torch.from_numpy(np.array([[[0]*seq_len]*num_heads] * batch_size)).float().to(features.device)

    return ad_out, torch.nn.BCELoss()(ad_out, label)
