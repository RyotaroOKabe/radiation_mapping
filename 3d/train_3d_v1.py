#%%

"""
original: train_torch_openmc_a3_sep_v2.py

"""

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
import tensorboardX as tbx

from torchvision import datasets, transforms
writer = tbx.SummaryWriter('runs')

from dataset_a3_v6 import *  #!20220508

from time_record import Timer

from unet_v1 import *  #!20220706

GPU_INDEX = 1#0
USE_CPU = False
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
        self.Wn = torch.nn.Parameter(data = torch.Tensor(1), requires_grad=True)
        self.weight = torch.as_tensor(filterdata.data.T)
        self.bias = torch.nn.Parameter(data=torch.Tensor(1,out_features), requires_grad=True)

    def forward(self,x):
        out = torch.matmul(x,self.weight)/self.Wn + self.bias
        return out


class Filterlayer2_ph(nn.Module):
    """docstring for Filterlayer"""
    def __init__(self, ph_num, th_num, filterdata):
        super(Filterlayer2_ph, self).__init__()
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


        out1 = (out1 -  out1.view(out1.shape[0], -1).mean(dim=1, keepdim=True).reshape((out1.shape[0], 1, 1)))/out1.view(out1.shape[0], -1).std(dim=1, keepdim=True).reshape((out1.shape[0], 1, 1))   # (N, 18, 40) #!20220711
        out2 = (out2 -  out2.view(out2.shape[0], -1).mean(dim=1, keepdim=True).reshape((out2.shape[0], 1, 1)))/out2.view(out1.shape[0], -1).std(dim=1, keepdim=True).reshape((out2.shape[0], 1, 1))

        out1_ph = (out1_ph -  out1_ph.mean(dim=1, keepdim=True).reshape((out1_ph.shape[0], 1)))/out1_ph.std(dim=1, keepdim=True).reshape((out1_ph.shape[0], 1))   # (N, 18, 40) #!20220711
        out1_th = (out1_th -  out1_th.mean(dim=1, keepdim=True).reshape((out1_th.shape[0], 1)))/out1_th.std(dim=1, keepdim=True).reshape((out1_th.shape[0], 1))   # (N, 18, 40) #!20220711
        out2_ph = (out2_ph -  out2_ph.mean(dim=1, keepdim=True).reshape((out2_ph.shape[0], 1)))/out2_ph.std(dim=1, keepdim=True).reshape((out2_ph.shape[0], 1))   # (N, 18, 40) #!20220711
        out2_th = (out2_th -  out2_th.mean(dim=1, keepdim=True).reshape((out2_th.shape[0], 1)))/out2_th.std(dim=1, keepdim=True).reshape((out2_th.shape[0], 1))   # (N, 18, 40) #!20220711

        out_ph = torch.cat([out1_ph,out2_ph],dim=1)
        out_th = torch.cat([out1_th,out2_th],dim=1)

        out_ph = out_ph.view(out_ph.shape[0], 2, self.ph_num)
        out_th = out_th.view(out_th.shape[0], 2, self.th_num)

        return out_ph


class Filterlayer2_th(nn.Module):
    """docstring for Filterlayer"""
    def __init__(self, ph_num, th_num, filterdata):
        super(Filterlayer2_th, self).__init__()
        #self.arg = arg
        self.ph_num = ph_num
        self.th_num = th_num
        
        self.Wn1 = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)
        self.Wn2 = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)

        temp = torch.torch.from_numpy(filterdata.data.reshape((2,self.th_num,self.ph_num,-1)))

        self.weight1 = torch.nn.Parameter(data=temp[0,:,:,:], requires_grad=True)
        self.weight2 = torch.nn.Parameter(data=temp[1,:,:,:], requires_grad=True)

        self.bias1 = torch.nn.Parameter(data=torch.zeros(1,self.th_num,self.ph_num), requires_grad=False)
        self.bias2 = torch.nn.Parameter(data=torch.zeros(1,self.th_num,self.ph_num), requires_grad=False)

        
        self.Wn1_test = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)
        self.Wn2_test = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)


    def forward(self,x):    # x: (N, 2, 18, 40)? or  (N, 1, 18, 40)? or  (N, 18, 40)? or (N, a^3)? or (N)
        #?out1 = torch.matmul(x,self.weight1)/self.Wn1 + self.bias1   # (N, 18, 40) x (1, 18?, 40?) >> #(N, 18, 40)
        #?out2 = torch.matmul(x,self.weight2)/self.Wn2 + self.bias2   #! (N, a^3) * (18, 40, a^3) >> (N, 18, 40)
        out1 = torch.einsum('bk,ijk->bij', x,self.weight1)/self.Wn1 + self.bias1   # (N, 18, 40) x (1, 18?, 40?) >> #(N, 18, 40)
        out2 = torch.einsum('bk,ijk->bij', x,self.weight2)/self.Wn2 + self.bias2   #! (N, a^3) * (18, 40, a^3) >> (N, 18, 40)

        out1 = (out1 -  out1.view(out1.shape[0], -1).mean(dim=1, keepdim=True).reshape((out1.shape[0], 1, 1)))/out1.view(out1.shape[0], -1).std(dim=1, keepdim=True).reshape((out1.shape[0], 1, 1))
        out2 = (out2 -  out2.view(out2.shape[0], -1).mean(dim=1, keepdim=True).reshape((out2.shape[0], 1, 1)))/out2.view(out1.shape[0], -1).std(dim=1, keepdim=True).reshape((out2.shape[0], 1, 1))

        out1_ph = torch.sum(torch.maximum(out1, 1.1*torch.ones_like(out1)), dim=1) # (N, ph_num)
        out1_th = torch.sum(torch.maximum(out1, 1.1*torch.ones_like(out1)), dim=2) # (N, th_num)
        out2_ph = torch.sum(torch.maximum(out2, 1.1*torch.ones_like(out2)), dim=1)
        out2_th = torch.sum(torch.maximum(out2, 1.1*torch.ones_like(out2)), dim=2)

        out1_ph = (out1_ph -  out1_ph.mean(dim=1, keepdim=True).reshape((out1_ph.shape[0], 1)))/out1_ph.std(dim=1, keepdim=True).reshape((out1_ph.shape[0], 1))
        out1_th = (out1_th -  out1_th.mean(dim=1, keepdim=True).reshape((out1_th.shape[0], 1)))/out1_th.std(dim=1, keepdim=True).reshape((out1_th.shape[0], 1))
        out2_ph = (out2_ph -  out2_ph.mean(dim=1, keepdim=True).reshape((out2_ph.shape[0], 1)))/out2_ph.std(dim=1, keepdim=True).reshape((out2_ph.shape[0], 1))
        out2_th = (out2_th -  out2_th.mean(dim=1, keepdim=True).reshape((out2_th.shape[0], 1)))/out2_th.std(dim=1, keepdim=True).reshape((out2_th.shape[0], 1))

        out_ph = torch.cat([out1_ph,out2_ph],dim=1) # (N, 2*ph_num)
        out_th = torch.cat([out1_th,out2_th],dim=1) # (N, 2*th_num)
        
        out_ph = out_ph.view(out_ph.shape[0], 2, self.ph_num)
        out_th = out_th.view(out_th.shape[0], 2, self.th_num)

        return out_th

class MyNet2_ph(nn.Module):
    def __init__(self, UNet_ph, ph_num, th_num, filterdata):
        super(MyNet2_ph, self).__init__()
        self.l1_ph = Filterlayer2_ph(ph_num, th_num, filterdata
        self.unet_ph = UNet_ph(in_channels=2, c1 = 32)

    def forward(self, x):
        ph = self.l1_ph(x)
        ph = self.unet_ph(ph)

        ph = ph.squeeze(1)
        out_ph = F.softmax(ph,dim=1)
        return out_ph

class MyNet2_th(nn.Module):
    def __init__(self, UNet_th, ph_num, th_num, filterdata):
        super(MyNet2_th, self).__init__()
        self.l1_th = Filterlayer2_th(ph_num, th_num, filterdata)
        self.unet_th = UNet_th(in_channels=2, c1 = 32)

    def forward(self, x):
        th = self.l1_th(x)
        th = self.unet_th(th)
        th = th.squeeze(1)
        out_th = F.softmax(th,dim=1)
        return out_th


class Model(object):
    def __init__(self, net_ph, net_th, loss_train_ph, loss_train_th, loss_val_ph, loss_val_th, reg=0.):
        super(Model, self).__init__()

        self.net_ph = net_ph
        self.net_th = net_th

        self.reg = reg

        self.loss_train_ph = loss_train_ph
        self.loss_train_th = loss_train_th

        if loss_val_ph is None:
            self.loss_val_ph=loss_train_ph
        else:
            self.loss_val_ph = loss_val_ph

        if loss_val_th is None:
            self.loss_val_th=loss_train_th
        else:
            self.loss_val_th = loss_val_th

    def train(self, optim_ph, optim_th, train, val, epochs,batch_size, acc_func_ph=None, acc_func_th=None, check_overfit_func=None,verbose=10, save_name='output_record'):
        net_ph = self.net_ph
        net_th = self.net_th
        loss_train_ph = self.loss_train_ph
        loss_val_ph = self.loss_val_ph
        loss_train_th = self.loss_train_th
        loss_val_th = self.loss_val_th
        t1=time.time()
        timer = Timer(['init','load data', 'forward', 'loss','cal reg', 'backward','optimizer step','eval'])    #!20220104

        train_loss_history_ph=[]
        val_loss_history_ph=[]
        train_loss_history_th=[]
        val_loss_history_th=[]

        if acc_func_ph is None:
            evaluation_ph=loss_val_ph
        else:
            evaluation_ph=acc_func_ph
        if acc_func_th is None:
            evaluation_th=loss_val_th
        else:
            evaluation_th=acc_func_th

        record_lines = []

        for i in range(epochs):
            times=int(math.ceil(train.data_size/float(batch_size)))
            datas=[]

            #net.train()
            net_ph.train()
            net_th.train()

            for j in range(times):

                timer.start('load data')

                data_x, data_y, data_y_ph, data_y_th=train.get_batch(batch_size)

                data_x = torch.from_numpy(data_x).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
                data_y = torch.from_numpy(data_y).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
                data_y_ph = torch.from_numpy(data_y_ph).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) 
                data_y_th = torch.from_numpy(data_y_th).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

                datas.append((data_x,data_y,data_y_ph,data_y_th))

                timer.end('load data');timer.start('forward')

                optim_ph.zero_grad()
                optim_th.zero_grad()

                output_ph = net_ph(data_x)
                output_th = net_th(data_x)
                timer.end('forward');timer.start('loss')

                loss_ph = loss_train_ph(data_y_ph, output_ph)
                loss_th = loss_train_th(data_y_th, output_th)

                timer.end('loss');timer.start('cal reg') 


                if self.reg:
                    reg_loss_ph=0.
                    nw_ph=0.
                    for param in net_ph.parameters():
                        reg_loss_ph +=  param.norm(2)**2/2.
                        nw_ph+=param.reshape(-1).shape[0]
                    reg_loss_ph = self.reg/2./nw_ph*reg_loss_ph
                    loss_ph += reg_loss_ph

                if self.reg:
                    reg_loss_th=0.
                    nw_th=0.
                    for param in net_th.parameters():
                        reg_loss_th +=  param.norm(2)**2/2.
                        nw_th+=param.reshape(-1).shape[0]

                    reg_loss_th = self.reg/2./nw_th*reg_loss_th
                    loss_th += reg_loss_th

                
                timer.end('cal reg');timer.start('backward')

                loss_ph.backward()
                loss_th.backward()

                timer.end('backward');timer.start('optimizer step')

                for param in net_ph.parameters():
                    pass
                for param in net_th.parameters():
                    pass

                optim_ph.step()
                optim_th.step()

                timer.end('optimizer step');

            train_loss_ph=0.
            val_loss_ph=0.
            train_loss_th=0.
            val_loss_th=0.

            timer.start('eval')

            net_ph.eval()
            net_th.eval()
            with torch.no_grad():
                for data in datas:
                    output_ph = net_ph(data[0])
                    output_th = net_th(data[0])
                    train_loss_ph+=evaluation_ph(data[2],output_ph)*data[0].shape[0]/train.data_size
                    train_loss_th+=evaluation_th(data[3],output_th)*data[0].shape[0]/train.data_size

                times=int(math.ceil(val.data_size/float(batch_size)))
                for j in range(times):
                    data=val.get_batch(batch_size,j)
                    data = (torch.as_tensor(data[0]),torch.as_tensor(data[1]),torch.as_tensor(data[2]),torch.as_tensor(data[3]))
                    output_ph = net_ph(data[0])
                    output_th = net_th(data[0])

                    val_loss_ph+=evaluation_ph(data[2],output_ph)*data[0].shape[0]/val.data_size
                    val_loss_th+=evaluation_th(data[3],output_th)*data[0].shape[0]/val.data_size

            writer.add_scalars('Training vs. Validation Loss', { 'Training_ph' : train_loss_ph, 'Validation_ph' : val_loss_ph,
                                                                'Training_th' : train_loss_th, 'Validation_th' : val_loss_th }, epochs)

            timer.end('eval')    #!20220104

            train_loss_history_ph.append(train_loss_ph)
            val_loss_history_ph.append(val_loss_ph)
            train_loss_history_th.append(train_loss_th)
            val_loss_history_th.append(val_loss_th)

            record_line = '%d\t%f\t%f\t%f\t%f'%(i,train_loss_ph,val_loss_ph,train_loss_th,val_loss_th)
            record_lines.append(record_line)

            if verbose and i%verbose==0:
                print('\t\tSTEP %d\t%f\t%f\t%f\t%f'%(i,train_loss_ph,val_loss_ph,train_loss_th,val_loss_th))

        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()

        t2=time.time()
        print('\t\tEPOCHS %d\t%f\t%f\t%f\t%f'%(epochs, train_loss_ph,val_loss_ph,train_loss_th,val_loss_th))
        print('\t\tFinished in %.1fs'%(t2-t1))
        self.train_loss_history_ph=train_loss_history_ph
        self.val_loss_history_ph=val_loss_history_ph
        self.train_loss_history_th=train_loss_history_th
        self.val_loss_history_th=val_loss_history_th

        text_file = open("save_record/" + save_name + ".txt", "w")
        for line in record_lines:
            text_file.write(line + "\n")
        text_file.close()


    def plot_train_curve(self, save_name): #!20220711
        fig = plt.figure(figsize=(20, 10), facecolor="white")
        ax1 = fig.add_subplot(121)

        ax1.plot([ls.cpu() for ls in self.train_loss_history_ph],label='training')
        ax1.plot([ls.cpu() for ls in self.val_loss_history_ph],label='validation')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Error')
        ax1.set_title('\u03C6')
        ax1.legend()
        ax2 = fig.add_subplot(122)
        ax2.plot([ls.cpu() for ls in self.train_loss_history_th],label='training')
        ax2.plot([ls.cpu() for ls in self.val_loss_history_th],label='validation')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Error')
        ax2.set_title('\u03B8')
        ax2.legend()
        fig.show()
        fig.savefig(fname="save_fig/train_" + save_name + ".png")
        plt.close()

    def save(self,name):
        data = {"tran_loss_hist_ph":self.train_loss_history_ph,"val_loss_hist_ph":self.val_loss_history_ph,
                "tran_loss_hist_th":self.train_loss_history_th,"val_loss_hist_th":self.val_loss_history_th}
        torch.save(data,name+'_log.pt')
        torch.save(self.net_ph,name+'_model_ph.pt')
        torch.save(self.net_th,name+'_model_th.pt')


    def plot_test(self,test,indx, save_name):

        test_x,test_y,test_y_ph,test_y_th=test.get_batch(1,indx)

        phi_list, theta_list = angles_lists(filterpath=filterpath)

        self.net_ph.eval()
        self.net_th.eval()
        with torch.no_grad():
            predict_test_ph = self.net_ph(torch.as_tensor(test_x))
            predict_test_th = self.net_th(torch.as_tensor(test_x))
            predict_test_ph = predict_test_ph.cpu().detach().numpy()
            predict_test_th = predict_test_th.cpu().detach().numpy()
        fig = plt.figure(figsize=(12, 6), facecolor='white')

        id_ref_ph = int(np.where(test_y_ph==test_y_ph.max())[1])
        id_ref_th = int(np.where(test_y_th==test_y_th.max())[1])
        ax11 = fig.add_subplot(1,2,1) #!!
        ax12 = fig.add_subplot(1,2,2)
        ax11.plot(phi_list, np.squeeze(test_y_ph),label='Simulated')
        ax12.plot(np.squeeze(test_y_th), theta_list,label='Simulated')
        pred_ph, pred_th = predict_test_ph, predict_test_th
        id_pred_ph = int(np.where(pred_ph==pred_ph.max())[1])
        id_pred_th = int(np.where(pred_th==pred_th.max())[1])
        ax11.plot(phi_list, np.squeeze(predict_test_ph),label='Predicted')
        ax11.set_title(f"[Simulated] \u03C6 = {phi_list[id_ref_ph]} deg ({id_ref_ph})\n[Predicted] \u03C6 = {phi_list[id_pred_ph]} deg ({id_pred_ph})")
        ax12.plot(np.squeeze(predict_test_th), theta_list,label='Predicted')
        ax12.set_title(f"[Simulated] \u03B8 = {theta_list[id_ref_th]} deg ({id_ref_th})\n[Predicted] \u03B8 = {theta_list[id_pred_th]} deg ({id_pred_th})")
        ax12.invert_yaxis()
        ax11.legend()
        ax12.legend()
        fig.show()
        fig.savefig(fname="save_fig/test_" + save_name + ".png")
        plt.close()


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
    #=========================================================
    save_name = "openmc_5cmxx3_ep100_bs256_20220714_v2.2" 
    #=========================================================
    path = 'openmc/discrete_data_20220706_5^3_v1'    #!20220630
    filterpath ='openmc/disc_filter_data_20220706_5^3_v1'
    filter_data2 = FilterData2(filterpath)
    UNet_ph = UNet
    UNet_th = UNet
    ph_num=32
    th_num=16
    net_ph = MyNet2_ph(UNet_ph, ph_num=ph_num, th_num=th_num, filterdata=filter_data2)
    net_ph = net_ph.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    net_th = MyNet2_th(UNet_th, ph_num=ph_num, th_num=th_num, filterdata=filter_data2)
    net_th = net_th.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
    kld_loss = torch.nn.KLDivLoss(size_average=None, reduction='batchmean')

    loss_train_ph = lambda  y, y_pred: emd_loss_ring(y, y_pred, r=2)
    loss_train_ph = nn.MSELoss() 
    loss_train_th = nn.MSELoss() 

    loss_val_ph = lambda y, y_pred: emd_loss_ring(y, y_pred, r=1).item()
    loss_val_ph = nn.MSELoss()
    loss_val_th = nn.MSELoss()

    model = Model(net_ph, net_th, loss_train_ph, loss_train_th, loss_val_ph, loss_val_th,reg=0.001)

    train_set,test_set=load_data(test_size=200,train_size=None,test_size_gen=None,output_fun=get_output,ph_num=ph_num, th_num=th_num, path=path,source_num=[1],prob=[1.],seed=None) #load_data(50, source_num=[1]) 

    optim_ph = torch.optim.Adam([
        {"params": net_ph.unet_ph.parameters()},
        {"params": net_ph.l1_ph.weight1, 'lr': 3e-5},
        {"params": net_ph.l1_ph.weight2, 'lr': 3e-5},
        {"params": net_ph.l1_ph.bias1, 'lr': 3e-5},
        {"params": net_ph.l1_ph.bias2, 'lr': 3e-5},
        {"params": net_ph.l1_ph.Wn2, 'lr': 3e-5},
        {"params": net_ph.l1_ph.Wn1, 'lr': 3e-5}
        ], lr=0.001)

    optim_th = torch.optim.Adam([
        {"params": net_th.unet_th.parameters()},
        {"params": net_th.l1_th.weight1, 'lr': 3e-5},
        {"params": net_th.l1_th.weight2, 'lr': 3e-5},
        {"params": net_th.l1_th.bias1, 'lr': 3e-5},
        {"params": net_th.l1_th.bias2, 'lr': 3e-5},
        {"params": net_th.l1_th.Wn2, 'lr': 3e-5},
        {"params": net_th.l1_th.Wn1, 'lr': 3e-5}
        ], lr=0.001)

    model.train(optim_ph,optim_th,train_set,test_set,epochs=1000,batch_size=256, acc_func_ph=None, acc_func_th=None, verbose=10, save_name=save_name)
    model.save('save_model/model_' + save_name)
    model.plot_train_curve(save_name=save_name)

    model.plot_test(test_set,indx=10, save_name=save_name)



# %%
