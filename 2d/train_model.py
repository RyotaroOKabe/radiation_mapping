#%%
"""
Created on 2022/07/16
original: train_torch_openmc_tetris_T_v2.py

@author: R.Okabe
"""

import os
import time
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import tensorboardX as tbx

from torchvision import datasets, transforms
writer = tbx.SummaryWriter('runs')
from utils.dataset import get_output, FilterData2, load_data
from utils.time_record import Timer
from utils.unet import *
from utils.model import MyNet2, Model
from utils.emd_ring_torch import emd_loss_ring

GPU_INDEX = 1#0
USE_CPU = False
if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")

DEFAULT_DTYPE = torch.double

#%%


a_num = 2
tetris_shape = 'L'
num_sources = 1
seg_angles = 64
epochs = 500
#=========================================================
save_name = f"openmc_{a_num}x{a_num}_{num_sources}src_{seg_angles}_ep{epochs}_bs256_20220822_v1.1"
#save_name = f"openmc_tetris{tetris_shape}_{num_sources}src_{seg_angles}_ep{epochs}_bs256_20220821_v1.1"
#=========================================================
# path = f'openmc/data_tetris{tetris_shape}_1src_64_data_20220821_v1.1'
# filterpath =f'openmc/filter_tetris{tetris_shape}_64_data_20220822_v1.1'
# path = 'openmc/data_2x2_1src_64_data_20220822_v1.1'  #!20220716
# filterpath ='openmc/filter_2x2_64_data_20220822_v1.1'    #!20220716
path = 'openmc/data_2x2_1src_64_data_20221003_v2.1'  #!20220716
filterpath ='openmc/filter_2x2_64_data_20221003_v2.1'    #!20220716
filter_data2 = FilterData2(filterpath)
test_size = 50  #! 0.9*train_size
k_fold = 5
print(save_name)
output_fun = get_output
net = MyNet2(seg_angles=seg_angles, filterdata=filter_data2)
net = net.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

kld_loss = torch.nn.KLDivLoss(size_average=None, reduction='batchmean')
loss_train = lambda  y, y_pred: emd_loss_ring(y, y_pred, r=2)
# loss_train = lambda y, y_pred: emd_loss_sinkhorn(y, y_pred, M2)
# loss_train = lambda y, y_pred: kld_loss(y_pred.log(),y
source_num, prob = [1 for _ in range(num_sources)], [1. for _ in range(num_sources)]

loss_val = lambda y, y_pred: emd_loss_ring(y, y_pred, r=1).item()
model = Model(net, loss_train, loss_val,reg=0.001)
train_set,test_set=load_data(test_size=test_size,train_size=None,test_size_gen=None,seg_angles=seg_angles,
                                output_fun=output_fun,path=path,source_num=source_num,prob=prob,seed=None)

optim = torch.optim.Adam([
    {"params": net.unet.parameters()},
    {"params": net.l1.weight1, 'lr': 3e-5},
    {"params": net.l1.weight2, 'lr': 3e-5},
    {"params": net.l1.bias1, 'lr': 3e-5},
    {"params": net.l1.bias2, 'lr': 3e-5},
    {"params": net.l1.Wn2, 'lr': 3e-5},
    {"params": net.l1.Wn1, 'lr': 3e-5}
    ], lr=0.001)

# model.train(optim,train_set,test_set,epochs,batch_size=256,acc_func=None, verbose=10, save_name=save_name)
model.train(optim,train_set,epochs,batch_size=256,split_fold=k_fold,acc_func=None, verbose=10, save_name=save_name)
model.save('save_model/model_' + save_name)

model.plot_train_curve(save_name=save_name)
model.plot_test(test_set,test_size,seg_angles=seg_angles,loss_fn=loss_val, save_name=save_name)
