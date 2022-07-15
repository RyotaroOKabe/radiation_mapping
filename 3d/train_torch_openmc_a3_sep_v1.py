#%%
import os
import time
#import tensorflow as tf
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
#from torch.utils.tensorboard import SummaryWriter
import tensorboardX as tbx

from torchvision import datasets, transforms
writer = tbx.SummaryWriter('runs')

#from dataset_20220508 import *  #!20220508
from dataset_a3_v6 import *  #!20220508

from time_record import Timer

#from unet import *  #!!
from unet_v1 import *  #!20220706

GPU_INDEX = 1#0
USE_CPU = False
# print torch.cuda.is_available()
if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda:%d"%GPU_INDEX) 
    torch.cuda.set_device(GPU_INDEX)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DEVICE = torch.device("cuda")
DEFAULT_DEVICE = "cuda"   #!20220704

DEFAULT_DTYPE = torch.double

def relu_cut(x): #!20220711
    return torch.maximum(x, 1.1)

class Filterlayer(nn.Module):
    """docstring for Filterlayer"""
    def __init__(self, out_features, filterdata):
        super(Filterlayer, self).__init__()
        #self.arg = arg
        self.Wn = torch.nn.Parameter(data = torch.Tensor(1), requires_grad=True)
        self.weight = torch.as_tensor(filterdata.data.T)
        # print self.weight.size()
        self.bias = torch.nn.Parameter(data=torch.Tensor(1,out_features), requires_grad=True)

    def forward(self,x):
        out = torch.matmul(x,self.weight)/self.Wn + self.bias
        return out

class Filterlayer2(nn.Module):
    """docstring for Filterlayer"""
    def __init__(self, ph_num, th_num, filterdata):
        super(Filterlayer2, self).__init__()
        #self.arg = arg
        self.ph_num = ph_num
        self.th_num = th_num
        
        self.Wn1 = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)
        self.Wn2 = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)

        temp = torch.torch.from_numpy(filterdata.data.reshape((2,self.th_num,self.ph_num,-1)))

        self.weight1 = torch.nn.Parameter(data=temp[0,:,:,:], requires_grad=True)#self.weight[:,:40] #(18, 40, 125) #!20220701
        self.weight2 = torch.nn.Parameter(data=temp[1,:,:,:], requires_grad=True)#self.weight[:,40:] #(18, 40, 125) #!20211230

        self.bias1 = torch.nn.Parameter(data=torch.zeros(1,self.th_num,self.ph_num), requires_grad=False)   # (1, 18, 40)  #!20220701
        self.bias2 = torch.nn.Parameter(data=torch.zeros(1,self.th_num,self.ph_num), requires_grad=False)

        
        self.Wn1_test = torch.nn.Parameter(data = torch.ones(1), requires_grad=True) #!20220305
        self.Wn2_test = torch.nn.Parameter(data = torch.ones(1), requires_grad=True) #!20220305


    def forward(self,x):    # x: (N, 2, 18, 40)? or  (N, 1, 18, 40)? or  (N, 18, 40)? or (N, a^3)? or (N)
        #?out1 = torch.matmul(x,self.weight1)/self.Wn1 + self.bias1   # (N, 18, 40) x (1, 18?, 40?) >> #(N, 18, 40)
        #?out2 = torch.matmul(x,self.weight2)/self.Wn2 + self.bias2   #! (N, a^3) * (18, 40, a^3) >> (N, 18, 40)
        out1 = torch.einsum('bk,ijk->bij', x,self.weight1)/self.Wn1 + self.bias1   # (N, 18, 40) x (1, 18?, 40?) >> #(N, 18, 40)
        out2 = torch.einsum('bk,ijk->bij', x,self.weight2)/self.Wn2 + self.bias2   #! (N, a^3) * (18, 40, a^3) >> (N, 18, 40)

        # print(out1.shape)
        # print((out1.view(out1.shape[0], -1).mean(dim=1, keepdim=True)).shape)
        # print((out1.view(out1.shape[0], -1).std(dim=1, keepdim=True)).shape)
        # print((out1.view(out1.shape[0], -1).std(dim=1, keepdim=True)).reshape((out1.shape[0], 1, 1)).shape)
        # print((out1.view(out1.shape[0], -1).std(dim=1, keepdim=True)).broadcast_to(out1.shape).shape)
        # print(((out1.view(out1.shape[0], -1).mean(dim=1)).broadcast_to(out1.shape)).shape)
        # print((out1 -  (out1.view(out1.shape[0], -1).mean(dim=1)).broadcast_to((out1.shape))).shape)

        # out1 = (out1 -  out1.view(out1.shape[0], -1).mean(dim=1))/out1.view(out1.shape[0], -1).std(dim=1)   # (N, 18, 40) #!20220711
        # out2 = (out2 -  out2.view(out2.shape[0], -1).mean(dim=1))/out2.view(out2.shape[0], -1).std(dim=1)

        out1 = (out1 -  out1.view(out1.shape[0], -1).mean(dim=1, keepdim=True).reshape((out1.shape[0], 1, 1)))/out1.view(out1.shape[0], -1).std(dim=1, keepdim=True).reshape((out1.shape[0], 1, 1))   # (N, 18, 40) #!20220711
        out2 = (out2 -  out2.view(out2.shape[0], -1).mean(dim=1, keepdim=True).reshape((out2.shape[0], 1, 1)))/out2.view(out1.shape[0], -1).std(dim=1, keepdim=True).reshape((out2.shape[0], 1, 1))
        
        # out1_ph = np.sum(torch.maximum(out1, 1.1), axis=1) # (N, ph_num)
        # out1_th = np.sum(torch.maximum(out1, 1.1), axis=2) # (N, th_num)
        # out2_ph = np.sum(torch.maximum(out2, 1.1), axis=1)
        # out2_th = np.sum(torch.maximum(out2, 1.1), axis=2)
        out1_ph = torch.sum(torch.maximum(out1, 1.1*torch.ones_like(out1)), dim=1) # (N, ph_num)
        out1_th = torch.sum(torch.maximum(out1, 1.1*torch.ones_like(out1)), dim=2) # (N, th_num)
        out2_ph = torch.sum(torch.maximum(out2, 1.1*torch.ones_like(out2)), dim=1)
        out2_th = torch.sum(torch.maximum(out2, 1.1*torch.ones_like(out2)), dim=2)

        # out1_ph = (out1_ph -  out1_ph.mean(dim=1))/out1_ph.std(dim=1) # (N, ph_num)
        # out1_th = (out1_th -  out1_th.mean(dim=1))/out1_th.std(dim=1) # (N, th_num)
        # out2_ph = (out2_ph -  out2_ph.mean(dim=1))/out2_ph.std(dim=1)
        # out2_th = (out2_th -  out2_th.mean(dim=1))/out2_th.std(dim=1)
        
        out1_ph = (out1_ph -  out1_ph.mean(dim=1, keepdim=True).reshape((out1_ph.shape[0], 1)))/out1_ph.std(dim=1, keepdim=True).reshape((out1_ph.shape[0], 1))   # (N, 18, 40) #!20220711
        out1_th = (out1_th -  out1_th.mean(dim=1, keepdim=True).reshape((out1_th.shape[0], 1)))/out1_th.std(dim=1, keepdim=True).reshape((out1_th.shape[0], 1))   # (N, 18, 40) #!20220711
        out2_ph = (out2_ph -  out2_ph.mean(dim=1, keepdim=True).reshape((out2_ph.shape[0], 1)))/out2_ph.std(dim=1, keepdim=True).reshape((out2_ph.shape[0], 1))   # (N, 18, 40) #!20220711
        out2_th = (out2_th -  out2_th.mean(dim=1, keepdim=True).reshape((out2_th.shape[0], 1)))/out2_th.std(dim=1, keepdim=True).reshape((out2_th.shape[0], 1))   # (N, 18, 40) #!20220711

        # out1_ph = relu_cut(out1_ph)
        # out1_th = relu_cut(out1_th)
        # out2_ph = relu_cut(out2_ph)
        # out2_th = relu_cut(out2_th)

        out_ph = torch.cat([out1_ph,out2_ph],dim=1) # (N, 2*ph_num)
        out_th = torch.cat([out1_th,out2_th],dim=1) # (N, 2*th_num)
        
        out_ph = out_ph.view(out_ph.shape[0], 2, self.ph_num)
        out_th = out_th.view(out_th.shape[0], 2, self.th_num)

        # out = torch.cat([out1,out2],dim=1) #(N, 36, 40) #! (N, 18, 40)(+)(N, 18, 40) >> (N, 36, 40)
        # out = out.view(out.shape[0], 2, self.th_num, self.ph_num)
        # return out
        return out_ph, out_th



# class MyNet2(nn.Module):  #!20220711 out
#     def __init__(self, UNet_option, ph_num, th_num, filterdata):   #!20220701
#         super(MyNet2, self).__init__()
#         self.l1 = Filterlayer2(ph_num, th_num, filterdata)#.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20220701
#         self.unet = UNet_option(c1 = 32)   #!20220701

    # def forward(self, x):   # x: (N, a^3) #!20220711 out
    #     x = self.l1(x)  # (N, a^3) * (18, 40, a^3) >> (N, 18, 40), (N, 18, 40)(+)(N, 18, 40) >> (N, 36, 40), (N, 36, 40) >> (N, 2, 18, 40)
    #     x = self.unet(x) # input=x:(N, 2, 18, 40) >> output=x: (N, 18, 40) or input=x:(N, 2, 40, 18) >> output=x: (N, 1, 40, 18)
    #     x = x.squeeze(1) # x: (N, 1, 40, 18) >> (N, 40, 18)
    #     out = F.softmax(x,dim=1)    # out: (N, 40, 18) >> (N, 18, 40)
    #     return out  # (N, 18, 40)

class MyNet2(nn.Module):
    def __init__(self, UNet_ph, UNet_th, ph_num, th_num, filterdata):   #!20220701
        super(MyNet2, self).__init__()
        self.l1 = Filterlayer2(ph_num, th_num, filterdata)#.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20220701
        self.unet_ph = UNet_ph(in_channels=2, c1 = 32)   #!20220711
        self.unet_th = UNet_th(in_channels=2, c1 = 32)   #!20220711

    def forward(self, x):
        ph, th = self.l1(x)
        #print(ph.shape)
        
        #ph = ph.transpose(1,2)
        
        ph = self.unet_ph(ph)
        th = self.unet_th(th)
        #print(ph)

        ph = ph.squeeze(1)
        th = th.squeeze(1)
        out_ph = F.softmax(ph,dim=1)
        out_th = F.softmax(th,dim=1)
        return out_ph, out_th


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        #self.fc1 = nn.Linear(in_features=100, out_features=80)
        self.l1 = Filterlayer(out_features=80, filterdata=filterdata)#.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        #self.filterbias = torch.zeros(1,80, requires_grad=True) -0.5
        self.fc2 = nn.Linear(in_features=80, out_features=40)
        self.fc3 = nn.Linear(in_features=40, out_features=40)
        self.fc4 = nn.Linear(in_features=40, out_features=40)
        self.fc5 = nn.Linear(in_features=40, out_features=40)
        self.fc6 = nn.Linear(in_features=40, out_features=40)
        self.fc7 = nn.Linear(in_features=40, out_features=40)

        self.fc8 = nn.Linear(in_features=40, out_features=40)
        
        
        #init weights
        #self.fc1.weight.data.normal_(0.0,1.0)
        self.fc2.weight.data.normal_(0.0,1.0)
        self.fc3.weight.data.normal_(0.0,1.0)
        self.fc4.weight.data.normal_(0.0,1.0)
        self.fc5.weight.data.normal_(0.0,1.0)
        self.fc6.weight.data.normal_(0.0,1.0)
        self.fc7.weight.data.normal_(0.0,1.0)
        self.fc8.weight.data.normal_(0.0,1.0)

        self.l1.bias.data.fill_(-0.5)
        self.l1.Wn.data.uniform_(5.,20.)
        #self.fc1.bias.data.normal_(0.0,1.0)
        self.fc2.bias.data.normal_(0.0,1.0)
        self.fc3.bias.data.normal_(0.0,1.0)
        self.fc4.bias.data.normal_(0.0,1.0)
        self.fc5.bias.data.normal_(0.0,1.0)
        self.fc6.bias.data.normal_(0.0,1.0)
        self.fc7.bias.data.normal_(0.0,1.0)

        self.fc8.bias.data.normal_(0.0,1.0)



    def forward(self, x):
        #print x.shape
        # print x.device
        out = F.elu(self.l1(x))
        #print out
        out = torch.sigmoid(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        out = torch.sigmoid(self.fc4(out))
        out = torch.sigmoid(self.fc5(out))
        out = torch.sigmoid(self.fc6(out))
        out = torch.sigmoid(self.fc7(out))

        #print out
        #out = F.relu(self.fc2(x))
        #out = F.sigmoid(self.fc3(out))
        #out = F.relu(self.fc3(x))
        out = self.fc8(out)
        #print out

        out = F.softmax(out,dim=1)
        #print out
        #raw_input()
        return out

class Model(object):
    #def __init__(self, net, loss_train, loss_val, reg=0.): #!20220711 out
    def __init__(self, net, loss_train_ph, loss_train_th, loss_val_ph, loss_val_th, reg=0.):  #!20220711
        super(Model, self).__init__()
        
        self.net = net

        self.reg = reg
        
        #self.loss_train = loss_train
        self.loss_train_ph = loss_train_ph  #!20220711
        self.loss_train_th = loss_train_th
        # if loss_val is None:
        #     self.loss_val=loss_train
        # else:
        #     self.loss_val = loss_val
        if loss_val_ph is None: #!20220711
            self.loss_val_ph=loss_train_ph
        else:
            self.loss_val_ph = loss_val_ph

        if loss_val_th is None: #!20220711
            self.loss_val_th=loss_train_th
        else:
            self.loss_val_th = loss_val_th
    
    
    #def train(self, optim, train, val, epochs,batch_size, acc_func=None, check_overfit_func=None,verbose=10, save_name='output_record'):
    def train(self, optim, train, val, epochs,batch_size, acc_func_ph=None, acc_func_th=None, check_overfit_func=None,verbose=10, save_name='output_record'):   #!20220711
        net = self.net
        # loss_train = self.loss_train
        # loss_val = self.loss_val
        loss_train_ph = self.loss_train_ph  #!20220711
        loss_val_ph = self.loss_val_ph
        loss_train_th = self.loss_train_th  #!20220711
        loss_val_th = self.loss_val_th
        
        t1=time.time()
        
        timer = Timer(['init','load data', 'forward', 'loss','cal reg', 'backward','optimizer step','eval'])    #!20220104
        
        # train_loss_history=[]
        # val_loss_history=[]
        train_loss_history_ph=[]   #!20220711
        val_loss_history_ph=[]
        train_loss_history_th=[]   #!20220711
        val_loss_history_th=[]

        # if acc_func is None:
        #     evaluation=loss_val
        # else:
        #     evaluation=acc_func
        if acc_func_ph is None: #!20220711
            evaluation_ph=loss_val_ph
        else:
            evaluation_ph=acc_func_ph
        if acc_func_th is None: #!20220711
            evaluation_th=loss_val_th
        else:
            evaluation_th=acc_func_th

        record_lines = []   #!20220126

        for i in range(epochs):
            times=int(math.ceil(train.data_size/float(batch_size)))
            #print 'train', times
            datas=[]

            net.train()

            
            

            for j in range(times):

                timer.start('load data')    #!20220104
                #Timer.start('load data')    #!20220104

                # data_x, data_y=train.get_batch_fixsource(batch_size,2)
                #data_x, data_y=train.get_batch(batch_size)  #!20220701 out
                data_x, data_y, data_y_ph, data_y_th=train.get_batch(batch_size)  #!20220701 
                
                
                

                #data_x = torch.as_tensor(data_x).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20211230
                #data_y = torch.as_tensor(data_y).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20211230
                data_x = torch.from_numpy(data_x).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20211230
                data_y = torch.from_numpy(data_y).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20211230
                data_y_ph = torch.from_numpy(data_y_ph).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20220701
                data_y_th = torch.from_numpy(data_y_th).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)    #!20220701

                #datas.append((data_x,data_y))   #!20220701 out
                datas.append((data_x,data_y,data_y_ph,data_y_th))   #!20220701  

                timer.end('load data');timer.start('forward')    #!20220104
                #Timer.end('load data');timer.start('forward')    #!20220104

                optim.zero_grad()

                
                #output = net(data_x)    #!20220711 out
                output_ph, output_th = net(data_x)    #!20220711
                #print(output)   #!20220305
                #print(data_y)   #!20220305
                #print(data_x)   #!20220305

                
                timer.end('forward');timer.start('loss')    #!20220104
                #Timer.end('forward');timer.start('loss')    #!20220104

                #=======The dimension check!=========#!20220508
                #print("=======The dimension check start!=========")
                #print("datay")
                #print(data_y)
                #print(data_y.shape)
                #print("y_sum_dim12.shape:", torch.sum(torch.sum(data_y,dim=2,keepdim=True),dim=1,keepdim=True).shape)
                #print("y_sum_dim12:", torch.sum(torch.sum(data_y,dim=2,keepdim=True),dim=1,keepdim=True))
                #print("y_sum_dim1:", torch.sum(data_y,dim=1,keepdim=True))
                #print("y_sum_all:", torch.sum(data_y))
                #print('----------')
                #print("output")
                #print(output)
                #print(output.shape)
                #print("output_sum_dim12.shape:", torch.sum(torch.sum(output,dim=2,keepdim=True),dim=1,keepdim=True).shape)
                #print("output_sum_dim12:", torch.sum(torch.sum(output,dim=2,keepdim=True),dim=1,keepdim=True))
                #print("output_sum_dim1:", torch.sum(output,dim=1,keepdim=True))
                #print("output_sum_all:", torch.sum(output))
                #sprint("=======The dimension check end!=========")
                #=======The dimension check!=========#!20220508
                
                #loss = loss_train(data_y, output) 
                #loss = loss_train(data_y.transpose(1,2), output.transpose(1,2))
                loss_ph = loss_train_ph(data_y_ph, output_ph)   #!20220711
                loss_th = loss_train_th(data_y_th, output_th)   #!20220711
                #loss = loss_train(10*17*data_y.transpose(1,2), 10**17*output.transpose(1,2))   #! Debug this part!!

                
                timer.end('loss');timer.start('cal reg')    #!20220104
                #Timer.end('loss');timer.start('cal reg')    #!20220104

                # if self.reg:    #!!
                #     reg_loss=0.
                #     nw=0.
                #     for param in net.parameters():
                #         # print param.shape
                #         # raw_input()
                #         reg_loss +=  param.norm(2)**2/2.    #! Debug here!!!
                #         nw+=param.reshape(-1).shape[0]


                #     reg_loss = self.reg/2./nw*reg_loss
                #     loss += reg_loss

                if self.reg:    #!20220711
                    reg_loss_ph=0.
                    reg_loss_th=0.
                    nw_ph=0.
                    nw_th=0.
                    for param in net.parameters():
                        # print param.shape
                        # raw_input()
                        reg_loss_ph +=  param.norm(2)**2/2.    #! Debug here!!!
                        reg_loss_th +=  param.norm(2)**2/2.
                        nw_ph+=param.reshape(-1).shape[0]
                        nw_th+=param.reshape(-1).shape[0]


                    reg_loss_ph = self.reg/2./nw_ph*reg_loss_ph
                    reg_loss_th = self.reg/2./nw_th*reg_loss_th
                    loss_ph += reg_loss_ph
                    loss_th += reg_loss_th

                
                timer.end('cal reg');timer.start('backward')    #!20220104
                #Timer.end('cal reg');timer.start('backward')    #!20220104

                #loss.backward()
                #loss = loss_ph + loss_th
                #(loss_ph + loss_th).backward()
                (loss_ph + loss_th).backward()

                timer.end('backward');timer.start('optimizer step')    #!20220104
                #Timer.end('backward');timer.start('optimizer step')    #!20220104
                

                for param in net.parameters():
                    #pass
                    #print param.shape,param.grad
                    #raw_input()
                    pass

                optim.step()

                timer.end('optimizer step');    #!20220104
                #Timer.end('optimizer step');    #!20220104
                
            #t2 = time.time(); print 'train data', t2-t0#; t1 = t2
            # train_loss=0.
            # val_loss=0.
            train_loss_ph=0.
            val_loss_ph=0.
            train_loss_th=0.
            val_loss_th=0.

            timer.start('eval')    #!20220104
            #Timer.start('eval')    #!20220104
            net.eval()
            with torch.no_grad():
                for data in datas:
                    #output = net(data[0])
                    output_ph, output_th = net(data[0])   #!20220711
                    #train_loss+=evaluation(data[1],output)*data[0].shape[0]/train.data_size
                    train_loss_ph+=evaluation_ph(data[2],output_ph)*data[0].shape[0]/train.data_size    #!20220711
                    train_loss_th+=evaluation_th(data[3],output_th)*data[0].shape[0]/train.data_size    #!20220711

                times=int(math.ceil(val.data_size/float(batch_size)))
                #print 'test', times
                for j in range(times):
                    data=val.get_batch(batch_size,j)

                    #data = (torch.as_tensor(data[0]),torch.as_tensor(data[1]))  #!20220701 out
                    data = (torch.as_tensor(data[0]),torch.as_tensor(data[1]),torch.as_tensor(data[2]),torch.as_tensor(data[3]))  #!20220701
                    #output = net(data[0])
                    output_ph, output_th = net(data[0])

                    #val_loss+=evaluation(data[1],output)*data[0].shape[0]/val.data_size
                    val_loss_ph+=evaluation_ph(data[2],output_ph)*data[0].shape[0]/val.data_size    #!20220711
                    val_loss_th+=evaluation_th(data[3],output_th)*data[0].shape[0]/val.data_size    #!20220711
            
            #writer.add_scalars('Training vs. Validation Loss', { 'Training' : train_loss, 'Validation' : val_loss }, epochs)     #!20220126__2
            writer.add_scalars('Training vs. Validation Loss', { 'Training_ph' : train_loss_ph, 'Validation_ph' : val_loss_ph,
                                                                'Training_th' : train_loss_th, 'Validation_th' : val_loss_th }, epochs)     #!20220711
            
            timer.end('eval')    #!20220104
            #Timer.end('eval')    #!20220104

            # train_loss_history.append(train_loss)
            # val_loss_history.append(val_loss)
            train_loss_history_ph.append(train_loss_ph)
            val_loss_history_ph.append(val_loss_ph)
            train_loss_history_th.append(train_loss_th)
            val_loss_history_th.append(val_loss_th)

            #record_line = '%d\t%f\t%f'%(i,train_loss,val_loss)    #!20220126
            record_line = '%d\t%f\t%f\t%f\t%f'%(i,train_loss_ph,val_loss_ph,train_loss_th,val_loss_th)    #!20220711
            record_lines.append(record_line)

            if verbose and i%verbose==0:
                #print('\t\tSTEP %d\t%f\t%f'%(i,train_loss,val_loss))
                print('\t\tSTEP %d\t%f\t%f\t%f\t%f'%(i,train_loss_ph,val_loss_ph,train_loss_th,val_loss_th))    #!20220711
                # print self.net.l1.Wn
                #timer.result(ratio=True)
                #print loss
                #raw_input()

        writer.export_scalars_to_json("./all_scalars.json")
            #writer.flush()       #!20220126__2
        writer.close()       #!20220126__2
        #writer.flush()       #!20220126__2

        t2=time.time()
        #print('\t\tEPOCHS %d\t%f\t%f'%(epochs, train_loss, val_loss))
        print('\t\tEPOCHS %d\t%f\t%f\t%f\t%f'%(epochs, train_loss_ph,val_loss_ph,train_loss_th,val_loss_th))    #!20220711
        print('\t\tFinished in %.1fs'%(t2-t1))
        # self.train_loss_history=train_loss_history
        # self.val_loss_history=val_loss_history
        self.train_loss_history_ph=train_loss_history_ph  #!20220711
        self.val_loss_history_ph=val_loss_history_ph
        self.train_loss_history_th=train_loss_history_th  #!20220711
        self.val_loss_history_th=val_loss_history_th
        
        #with open("save_record/" + save_name + ".txt", "w") as text_file:   #!20220126
        text_file = open("save_record/" + save_name + ".txt", "w")
        for line in record_lines:
            text_file.write(line + "\n")   #!20220126
        text_file.close()


    # def plot_train_curve(self):
    #     #plt.figure()
    #     plt.figure(facecolor="white")
    #     plt.plot(self.train_loss_history,label='training')
    #     plt.plot(self.val_loss_history,label='validation')
    #     plt.xlabel('Steps')
    #     plt.ylabel('Error')
    #     plt.legend()

    def plot_train_curve(self): #!20220711
        #plt.figure()
        fig = plt.figure(figsize=(10, 20), facecolor="white")
        ax1 = fig.add_subplot(121)
        ax1.plot(self.train_loss_history_ph,label='training')
        ax1.plot(self.val_loss_history_ph,label='validation')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Error')
        ax1.set_title('\u03C6')
        ax1.legend()
        ax2 = fig.add_subplot(122)
        # ax2.plot(self.train_loss_history_th,label='training')
        # ax2.plot(self.val_loss_history_th,label='validation')
        ax2.plot([ls.cpu() for ls in self.train_loss_history_th],label='training')
        ax2.plot([ls.cpu() for ls in self.val_loss_history_th],label='validation')
        #ax2.plot(self.val_loss_history_th.cpu(),label='validation')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Error')
        ax2.set_title('\u03B8')
        ax2.legend()

    def save(self,name):
        #data = {"tran_loss_hist":self.train_loss_history,"val_loss_hist":self.val_loss_history}
        data = {"tran_loss_hist_ph":self.train_loss_history_ph,"val_loss_hist_ph":self.val_loss_history_ph,
                "tran_loss_hist_th":self.train_loss_history_th,"val_loss_hist_th":self.val_loss_history_th} #!20220711
        torch.save(data,name+'_log.pt')
        torch.save(self.net,name+'_model.pt')


    def plot_test(self,test,indx):

        #test_x,test_y=test.get_batch(1,indx)    #!20220701 out
        test_x,test_y,test_y_ph,test_y_th=test.get_batch(1,indx)    #!20220701

        phi_list, theta_list = angles_lists(filterpath=filterpath)
        
        #test_y = 
        #test_indx=355
        #print y_test[test_indx,:]
        #predict_test=sess.run(self.outputs,feed_dict={xs:test_x,ys:test_y})
        self.net.eval()
        with torch.no_grad():
            #predict_test = self.net(torch.as_tensor(test_x)).detach().numpy()
            predict_test_ph, predict_test_th = self.net(torch.as_tensor(test_x)).detach().numpy()   #!20220711

        #fig = plt.figure(figsize=(50, 14))  #!20220701
        fig = plt.figure(figsize=(50, 25), facecolor='white')  #!20220707
        
        #! 20220701
        # do0 = test_y
        # id_t_0, id_p_0 = [int(np.where(do0==do0.max())[i]) for i in range(2)]
        # ax2 = fig.add_subplot(211)
        # ax2.imshow(do0, extent=[phi_list[0],phi_list[-1],theta_list[-1],theta_list[0]])
        # ax2.set_title(f"[Test Ref]\nPeak: \u03C6 = {phi_list[id_p_0]} deg ({id_p_0}) /  \u03B8 = {theta_list[id_t_0]} deg ({id_t_0})")
        # ax2.set_ylabel('\u03B8 [deg]', fontsize = 16)

        # do1 = predict_test
        # id_t_1, id_p_1 = [int(np.where(do1==do1.max())[i]) for i in range(2)]
        # ax3 = fig.add_subplot(212)
        # ax3.imshow(do1, extent=[phi_list[0],phi_list[-1],theta_list[-1],theta_list[0]])
        # ax3.set_title(f"[Prediction]\nPeak: \u03C6 = {phi_list[id_p_1]} deg ({id_p_1}) /  \u03B8 = {theta_list[id_t_1]} deg ({id_t_1})")
        # ax3.set_ylabel('\u03B8 [deg]', fontsize = 16)
        # fig.subplots_adjust(left=0.1,
        #             bottom=0.1, 
        #             right=0.9, 
        #             top=0.9, 
        #             wspace=0.3, 
        #             hspace=0.4)

        id_ref_ph = int(np.where(test_y_ph==test_y_ph.max())[0])
        id_ref_th = int(np.where(test_y_th==test_y_th.max())[0])
        ax11 = fig.add_subplot(2,2,1) #!!
        ax12 = fig.add_subplot(2,2,2)
        ax11.plot(phi_list, test_y_ph)
        ax11.set_title(f"[Simulated]\nPeak: \u03C6 = {phi_list[id_ref_ph]} deg ({id_ref_ph})")
        ax12.plot(test_y_th, theta_list)
        ax12.set_title(f"[Simulated]\nPeak: \u03B8 = {theta_list[id_ref_th]} deg ({id_ref_th})")

        pred_ph, pred_th = predict_test_ph.cpu().numpy(), predict_test_th.cpu().numpy()
        id_pred_ph = int(np.where(pred_ph==pred_ph.max())[0])
        id_pred_th = int(np.where(pred_th==pred_th.max())[0])
        ax21 = fig.add_subplot(2,2,3) #!!
        ax22 = fig.add_subplot(2,2,4)
        ax21.plot(phi_list, predict_test_ph)
        ax21.set_title(f"[Predicted]\nPeak: \u03C6 = {phi_list[id_pred_ph]} deg ({id_pred_ph})")
        ax22.plot(predict_test_th, theta_list)
        ax22.set_title(f"[Predicted]\nPeak: \u03B8 = {theta_list[id_pred_th]} deg ({id_pred_th})")





        #fig.patch.set_facecolor('white')
        ###fig.show()
        ###fig.savefig("save_fig/test_result.png")
        ###plt.close()
        #plt.figure()
        #plt.plot(np.linspace(-180,180,41)[0:40],test_y[0])
        #plt.plot(np.linspace(-180,180,41)[0:40],predict_test[0])
        #plt.legend(['Real','Prediction'])
        #plt.xlabel('deg')
        #plt.xlim([-180,180])
        #plt.xticks([-180,-135,-90,-45,0,45,90,135,180])

from pyemd import emd
n=40
M1=np.zeros([n,n])
M2=np.zeros([n,n])
for i in range(n):
    for j in range(n):
        #M1[i,j]=abs(i-j)
        M1[i,j]=min(abs(i-j),j+n-i,i+n-j)#**2
        M2[i,j]=min(abs(i-j),j+n-i,i+n-j)**2

from emd_ring_torch_3d_v1 import emd_loss_ring, emd_loss_ring_3d

from emd_sinkhorn_torch import emd_pop_zero_batch, sinkhorn_torch

def emd_ring(y_preds,ys,M):
    loss=0.
    for i in range(y_preds.shape[0]):
        loss+= emd(y_preds[i,:],ys[i,:],M1)
    return loss/y_preds.shape[0]

def emd_loss_sinkhorn(y, y_pred, M, reg=1.5, iter_num=80):
    
    y_new,Ms=emd_pop_zero_batch(np.asarray(y.cpu()),M)
    y_new=torch.as_tensor(y_new)
    Ms=torch.as_tensor(Ms)
    loss=sinkhorn_torch(y_new,y_pred,Ms,reg,numItermax=iter_num)

    return loss

timer = Timer(['init','load data', 'forward', 'loss','cal reg', 'backward','optimizer step','eval'])

#%%

if __name__ == '__main__':
    #print filterdata.data.shape
    #print('ws_point0')   #!20220303
    #=========================================================
    save_name = "openmc_5cmxx3_ep30_bs256_20220712_v1.2"      #!20220126
    #=========================================================
    path = 'openmc/discrete_data_20220706_5^3_v1'    #!20220630
    filterpath ='openmc/disc_filter_data_20220706_5^3_v1'
    filter_data2 = FilterData2(filterpath)
    #UNet_option = UNet_3D_2
    UNet_ph = UNet
    UNet_th = UNet
    ph_num=32
    th_num=16
    
    #net = MyNet2(out_features=80, filterdata=filter_data2)  #!20220701
    # (self, UNet_ph, UNet_th, ph_num, th_num, filterdata)
    #net = MyNet2(UNet_option, ph_num=ph_num, th_num=th_num, filterdata=filter_data2)  #!20220701
    net = MyNet2(UNet_ph, UNet_th, ph_num=ph_num, th_num=th_num, filterdata=filter_data2)   #!20220711

    net = net.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

    
    #kld_loss = torch.nn.KLDivLoss(reduction='batchmean')    #!20211230      #?tentative
    #kld_loss = torch.nn.KLDivLoss(size_average=None, reduce=None)    #!20211230      #?tentative
    kld_loss = torch.nn.KLDivLoss(size_average=None, reduction='batchmean')    #!20220104

    #os.nice(19)
    #loss_train = lambda y_pred, y: torch.sum(torch.norm(y_pred-y, p=2, dim=1))
    #loss_train = lambda  y, y_pred: emd_loss_ring_3d(y, y_pred, r=2)   #! debug this part!!
    loss_train_ph = lambda  y, y_pred: emd_loss_ring(y, y_pred, r=2)    #!20220711
    loss_train_th = nn.MSELoss() #lambda  y, y_pred: emd_loss_ring(y, y_pred, r=2)    #!20220711
    # loss_train = lambda y, y_pred: emd_loss_sinkhorn(y, y_pred, M2)
    # loss_train = lambda y, y_pred: kld_loss(y_pred.log(),y)
    
    #loss_val = lambda y_pred, y: emd_ring(np.asarray(y_pred),  np.asarray(y), M1)
    #loss_val = lambda y, y_pred: emd_loss_ring_3d(y, y_pred, r=1).item()
    loss_val_ph = lambda y, y_pred: emd_loss_ring(y, y_pred, r=1).item()    #!20220711
    loss_val_th = nn.MSELoss()#.item()  #lambda y, y_pred: emd_loss_ring(y, y_pred, r=1).item()    #!20220711

    # (self, net, loss_train_ph, loss_train_th, loss_val_ph, loss_val_th, reg=0.)
    #model = Model(net, loss_train, loss_val,reg=0.001)
    model = Model(net, loss_train_ph, loss_train_th, loss_val_ph, loss_val_th,reg=0.001)  #!20220711

    train_set,test_set=load_data(test_size=30,train_size=None,test_size_gen=None,output_fun=get_output,ph_num=ph_num, th_num=th_num, path=path,source_num=[1],prob=[1.],seed=None) #load_data(50, source_num=[1]) 
    #train_set,test_set=load_data(600, source_num=[1])   #!20220508
    #print(train_set)
    #print(test_set) #!20220303


    
    # optim = torch.optim.Adam(net.parameters(), lr=0.001)

    # l1_params = list(map(id, net.l1.parameters()))
    # conv4_params = list(map(id, net.conv4.parameters()))
    # base_params = filter(lambda p: id(p) not in l1_params,
                         # net.parameters())
    optim = torch.optim.Adam([
        #{"params": net.unet.parameters()},
        {"params": net.unet_ph.parameters()},  #!20220711
        #{"params": net.unet_ph.parameters()},
        {"params": net.l1.weight1, 'lr': 3e-5},
        {"params": net.l1.weight2, 'lr': 3e-5},
        {"params": net.l1.bias1, 'lr': 3e-5},
        {"params": net.l1.bias2, 'lr': 3e-5},
        {"params": net.l1.Wn2, 'lr': 3e-5},
        {"params": net.l1.Wn1, 'lr': 3e-5}
        ], lr=0.001)

    #model.train(optim,train_set,test_set,epochs=100,batch_size=256, acc_func=None, verbose=10, save_name=save_name)    #!20220126
    model.train(optim,train_set,test_set,epochs=30,batch_size=256, acc_func_ph=None, acc_func_th=None, verbose=10, save_name=save_name)   #!20220711

    #model.save('test8')
    #model.save('model_' + save_name)     #!20220126
    model.save('save_model/model_' + save_name)     #!20220711

    model.plot_train_curve()
    plt.show()
    #plt.savefig(fname="save_fig/train_6000_20220107.png")    #!20220104
    plt.savefig(fname="save_fig/train_" + save_name + ".png")    #!20220126
    plt.close()
    
    model.plot_test(test_set,indx=10)
    plt.show()
    #plt.savefig(fname="save_fig/train_6000_20220107.png")    #!20220104
    plt.savefig(fname="save_fig/test_" + save_name + ".png")    #!20220126
    plt.close()
    #raw_input()    #!20220104
    #plt.show()


# %%
