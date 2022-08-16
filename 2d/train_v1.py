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

from dataset_v1 import *  #!20220716

from time_record import Timer

from unet import *

GPU_INDEX = 1#0
USE_CPU = False
if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")

DEFAULT_DTYPE = torch.double

 

class Filterlayer(nn.Module):
    """docstring for Filterlayer"""
    def __init__(self, out_features, filterdata):
        super(Filterlayer, self).__init__()
        self.Wn = torch.nn.Parameter(data = torch.Tensor(1), requires_grad=True)
        self.weight = torch.as_tensor(filterdata.data.T)
        self.bias = torch.nn.Parameter(data=torch.Tensor(1,out_features), requires_grad=True)

    def forward(self,x):
        out = torch.matmul(x,self.weight)/self.Wn + self.bias
        return out

class Filterlayer2(nn.Module):
    """docstring for Filterlayer"""
    def __init__(self, seg_angles, out_features, filterdata):
        super(Filterlayer2, self).__init__()
        self.Wn1 = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)
        self.Wn2 = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)
        temp = torch.torch.from_numpy(filterdata.data.reshape((2,seg_angles,-1)))
        self.weight1 = torch.nn.Parameter(data=torch.t(temp[0,:,:]), requires_grad=True)
        self.weight2 = torch.nn.Parameter(data=torch.t(temp[1,:,:]), requires_grad=True)

        self.bias1 = torch.nn.Parameter(data=torch.zeros(1,out_features//2), requires_grad=True)
        self.bias2 = torch.nn.Parameter(data=torch.zeros(1,out_features//2), requires_grad=True)
        
        self.Wn1_test = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)
        self.Wn2_test = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)

    def forward(self,x):
        out1 = torch.matmul(x,self.weight1)/self.Wn1 + self.bias1
        out2 = torch.matmul(x,self.weight2)/self.Wn2 + self.bias2
        out = torch.cat([out1,out2],dim=1)
        out = out.view(out.shape[0], 2, out.shape[1]//2)
        return out

     


class MyNet2(nn.Module):
    def __init__(self, seg_angles, filterdata): #!20220729
        super(MyNet2, self).__init__()
        self.l1 = Filterlayer2(seg_angles=seg_angles, out_features=2*seg_angles, filterdata=filterdata)
        self.unet = UNet(c1 = 32)

    def forward(self, x):
        x = self.l1(x)
        x = self.unet(x)
        x = x.squeeze(1)
        out = F.softmax(x,dim=1)
        return out

class Model(object):
    def __init__(self, net, loss_train, loss_val, reg=0.):
        super(Model, self).__init__()
        self.net = net
        self.reg = reg
        self.loss_train = loss_train
        if loss_val is None:
            self.loss_val=loss_train
        else:
            self.loss_val = loss_val

    def train(self, optim, train, val, epochs,batch_size, acc_func=None, check_overfit_func=None,verbose=10, save_name='output_record'):
        net = self.net
        loss_train = self.loss_train
        loss_val = self.loss_val
        t1=time.time()
        timer = Timer(['init','load data', 'forward', 'loss','cal reg', 'backward','optimizer step','eval'])    #!20220104
        train_loss_history=[]
        val_loss_history=[]

        if acc_func is None:
            evaluation=loss_val
        else:
            evaluation=acc_func
        record_lines = []

        for i in range(epochs):
            times=int(math.ceil(train.data_size/float(batch_size)))
            datas=[]
            net.train()
            for j in range(times):
                timer.start('load data')    #!20220104
                data_x, data_y=train.get_batch(batch_size)
                data_x = torch.from_numpy(data_x).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
                data_y = torch.from_numpy(data_y).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
                datas.append((data_x,data_y))
                timer.end('load data');timer.start('forward')
                optim.zero_grad()
                output = net(data_x)
                timer.end('forward');timer.start('loss')
                loss = loss_train(data_y, output)
                timer.end('loss');timer.start('cal reg')
                if self.reg:
                    reg_loss=0.
                    nw=0.
                    for param in net.parameters():
                        reg_loss +=  param.norm(2)**2/2.
                        nw+=param.reshape(-1).shape[0]

                    reg_loss = self.reg/2./nw*reg_loss
                    loss += reg_loss

                timer.end('cal reg');timer.start('backward')

                loss.backward()

                timer.end('backward');timer.start('optimizer step')

                for param in net.parameters():
                    pass

                optim.step()

                timer.end('optimizer step');

            train_loss=0.
            val_loss=0.

            timer.start('eval')
            net.eval()
            with torch.no_grad():
                for data in datas:
                    output = net(data[0])
                    train_loss+=evaluation(data[1],output)*data[0].shape[0]/train.data_size

                times=int(math.ceil(val.data_size/float(batch_size)))
                for j in range(times):
                    data=val.get_batch(batch_size,j)

                    data = (torch.as_tensor(data[0]),torch.as_tensor(data[1]))
                    output = net(data[0])

                    val_loss+=evaluation(data[1],output)*data[0].shape[0]/val.data_size
            
            writer.add_scalars('Training vs. Validation Loss', { 'Training' : train_loss, 'Validation' : val_loss }, epochs)     #!20220126__2
            
            timer.end('eval')

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            
            record_line = '%d\t%f\t%f'%(i,train_loss,val_loss)
            record_lines.append(record_line)

            if verbose and i%verbose==0:
                print('\t\tSTEP %d\t%f\t%f'%(i,train_loss,val_loss))

        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()

        t2=time.time()
        print('\t\tEPOCHS %d\t%f\t%f'%(epochs, train_loss, val_loss))
        print('\t\tFinished in %.1fs'%(t2-t1))
        self.train_loss_history=train_loss_history
        self.val_loss_history=val_loss_history

        text_file = open("save_record/" + save_name + ".txt", "w")
        for line in record_lines:
            text_file.write(line + "\n")
        text_file.close()

    def plot_train_curve(self, save_name):
        figpath = "save_fig/" + save_name
        if not os.path.isdir(figpath):
            os.mkdir(figpath)
        fig = plt.figure(figsize=(6, 6), facecolor="white")
        ax1 = fig.add_subplot(111)
        ax1.plot(self.train_loss_history,label='training')
        ax1.plot(self.val_loss_history,label='validation')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Error')
        ax1.legend()
        fig.show()
        fig.savefig(fname=f"{figpath}/train.png")
        fig.savefig(fname=f"{figpath}/train.pdf")
        plt.close()


    def save(self,name):
        data = {"tran_loss_hist":self.train_loss_history,"val_loss_hist":self.val_loss_history}
        torch.save(data,name+'_log.pt')
        torch.save(self.net,name+'_model.pt')

    def plot_test(self,test,test_size, seg_angles,loss_fn, save_name):

        figpath = "save_fig/" + save_name
        if not os.path.isdir(figpath):
            os.mkdir(figpath)

        total_loss = 0
        for indx in range(test_size):
            test_x,test_y=test.get_batch(1,indx)

            self.net.eval()
            with torch.no_grad():
                predict_test = self.net(torch.as_tensor(test_x)).cpu().detach().numpy()

            pred_loss = loss_train(torch.Tensor(test_y[0]).reshape((1, -1)), torch.Tensor(predict_test[0]).reshape((1, -1)))
            # print('pred_loss: ', pred_loss)
            total_loss += pred_loss.item()
            fig = plt.figure(figsize=(6, 6), facecolor='white')  #!20220707
            ax1 = fig.add_subplot(1,1,1)
            #plt.figure()
            ax1.plot(np.linspace(-180,180,seg_angles+1)[0:seg_angles],test_y[0],label='Simulated')
            ax1.plot(np.linspace(-180,180,seg_angles+1)[0:seg_angles],predict_test[0],label='Predicted')
            #ax1.legend(['Real','Prediction'])
            ax1.legend()
            ax1.set_xlabel('deg')
            ax1.set_xlim([-180,180])
            ax1.set_xticks([-180,-135,-90,-45,0,45,90,135,180])
            ax1.set_title(f'Loss: {pred_loss.item()}')  #!20220813
            fig.show()
            fig.savefig(fname=f"{figpath}/test_{indx}.png")
            fig.savefig(fname=f"{figpath}/test_{indx}.pdf")
            plt.close()
        loss_avg = total_loss/test_size
        print('Average loss dist: ', loss_avg)


from pyemd import emd
n=40
M1=np.zeros([n,n])
M2=np.zeros([n,n])
for i in range(n):
    for j in range(n):
        M1[i,j]=min(abs(i-j),j+n-i,i+n-j)#**2
        M2[i,j]=min(abs(i-j),j+n-i,i+n-j)**2

from emd_ring_torch import emd_loss_ring

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
    a_num = 2
    num_sources = 2
    seg_angles = 128
    epochs = 10000
    #=========================================================
    save_name = f"openmc_{a_num}x{a_num}_{num_sources}src_{seg_angles}_ep{epochs}_bs256_20220815_v1.1"
    #=========================================================
    path = 'openmc/discrete_2x2_2src_128_data_20220811_v1.1'
    filterpath ='openmc/disc_filter_2x2_128_data_20220813_v1.1'
    filter_data2 = FilterData2(filterpath)
    test_size = 50
    print(save_name)
    net = MyNet2(seg_angles=seg_angles, filterdata=filter_data2)
    net = net.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    kld_loss = torch.nn.KLDivLoss(size_average=None, reduction='batchmean')
    loss_train = lambda  y, y_pred: emd_loss_ring(y, y_pred, r=2)
    loss_val = lambda y, y_pred: emd_loss_ring(y, y_pred, r=1).item()
    model = Model(net, loss_train, loss_val,reg=0.001)
    train_set,test_set=load_data(test_size=test_size,train_size=None,test_size_gen=None,seg_angles=seg_angles,output_fun=get_output_2source,path=path,source_num=[1,1],prob=[1., 1.],seed=None)

    optim = torch.optim.Adam([
        {"params": net.unet.parameters()},
        {"params": net.l1.weight1, 'lr': 3e-5},
        {"params": net.l1.weight2, 'lr': 3e-5},
        {"params": net.l1.bias1, 'lr': 3e-5},
        {"params": net.l1.bias2, 'lr': 3e-5},
        {"params": net.l1.Wn2, 'lr': 3e-5},
        {"params": net.l1.Wn1, 'lr': 3e-5}
        ], lr=0.001)

    model.train(optim,train_set,test_set,epochs,batch_size=256,acc_func=None, verbose=10, save_name=save_name)
    model.save('save_model/model_' + save_name)

    model.plot_train_curve(save_name=save_name)
    model.plot_test(test_set,test_size,seg_angles=seg_angles,loss_fn=loss_val, save_name=save_name)

# %%
