#%%
import os
import time
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
from utils.utils import *

GPU_INDEX = 1#0
USE_CPU = False
if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")

DEFAULT_DTYPE = torch.double
colors_parameters = {'array_hex':'#EEAD0E', 'pred_hex':'#CA6C4A' , 'real_hex': '#77C0D2'}
colors_max, pred_rgb, real_rgb = [hex2rgb(colors_parameters[l]) for l in ['array_hex', 'pred_hex', 'real_hex']]
N = 255

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

class Filterlayer1(nn.Module):
    """docstring for Filterlayer"""
    def __init__(self, seg_angles, out_features, filterdata):
        super(Filterlayer1, self).__init__()
        self.Wn1 = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)
        temp = torch.torch.from_numpy(filterdata.data.reshape((1,seg_angles,-1)))
        self.weight1 = torch.nn.Parameter(data=torch.t(temp[0,:,:]), requires_grad=True)
        self.bias1 = torch.nn.Parameter(data=torch.zeros(1,out_features), requires_grad=True)
        self.Wn1_test = torch.nn.Parameter(data = torch.ones(1), requires_grad=True)

    def forward(self,x):
        out1 = torch.matmul(x,self.weight1)/self.Wn1 + self.bias1
        out = out1 
        out = out.view(out.shape[0], 1, out.shape[1]//1)
        return out

class MyNet2(nn.Module):
    def __init__(self, seg_angles, filterdata):
        super(MyNet2, self).__init__()
        self.l1 = Filterlayer2(seg_angles=seg_angles, out_features=2*seg_angles, filterdata=filterdata)
        self.unet = UNet(c1 = 32)

    def forward(self, x):
        x = self.l1(x)
        x = self.unet(x)
        x = x.squeeze(1)
        out = F.softmax(x,dim=1)
        return out

class MyNet1(nn.Module):
    def __init__(self, seg_angles, filterdata):
        super(MyNet1, self).__init__()
        self.l1 = Filterlayer1(seg_angles=seg_angles, out_features=seg_angles, filterdata=filterdata)
        self.unet = UNet(in_channels=1, c1 = 32)

    def forward(self, x):
        x = self.l1(x)
        x = self.unet(x)
        x = x.squeeze(1)
        out = F.softmax(x,dim=1)
        return out

def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))

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

    def train(self, optim, train_set, test_set, epochs,batch_size, split_fold, acc_func=None, check_overfit_func=None,verbose=10, save_dir='./save/training', save_file='output_record'):
        net = self.net
        loss_train = self.loss_train
        loss_val = self.loss_val
        t1=time.time()
        timer = Timer(['init','load data', 'forward', 'loss','cal reg', 'backward','optimizer step','eval'])
        save_name = os.path.join(save_dir, save_file)
        if not os.path.isdir(save_name):
            os.mkdir(save_name)
        record_header = '%s\t%s\t%s\t%s'%('Epochs',"train_loss","val_loss", "Time")
        train_loss_history=[]
        val_loss_history=[]
        if acc_func is None:
            evaluation=loss_val
        else:
            evaluation=acc_func
        record_lines = []
        
        checkpoint_generator = loglinspace(0.3, 5)
        checkpoint = next(checkpoint_generator)
        
        for i in range(epochs):
            tr_set, va_set = train_set.split(split_fold=split_fold, step=i)
            times=int(math.ceil(tr_set.data_size/float(batch_size)))
            datas=[]
            net.train()
            for j in range(times):
                timer.start('load data') 
                data_x, data_y, data_z=tr_set.get_batch(batch_size)
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
                timer.end('optimizer step')
            train_loss=0.
            val_loss=0.
            timer.start('eval')
            net.eval()
            with torch.no_grad():
                for data in datas:
                    output = net(data[0])
                    train_loss+=evaluation(data[1],output)*data[0].shape[0]/tr_set.data_size

                times=int(math.ceil(va_set.data_size/float(batch_size)))
                for j in range(times):
                    data=va_set.get_batch(batch_size,j)
                    data = (torch.as_tensor(data[0]),torch.as_tensor(data[1]))
                    output = net(data[0])
                    val_loss+=evaluation(data[1],output)*data[0].shape[0]/va_set.data_size
            writer.add_scalars('Training vs. Validation Loss', { 'Training' : train_loss, 'Validation' : val_loss }, epochs)
            timer.end('eval')
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            t2=time.time()
            record_line = '%d\t%f\t%f\t%f'%(i,train_loss,val_loss,t2-t1)
            record_lines.append(record_line)
            
            if i == checkpoint:
                print('\t\tSTEP %d\t%f\t%f'%(i,train_loss,val_loss))
                self.train_loss_history=train_loss_history
                self.val_loss_history=val_loss_history
                checkpoint = next(checkpoint_generator)

                self.plot_train_curve(save_name)
                self.save('save/models/' + save_file)
                loss_profile = self.plot_test(test_set,loss_fn=loss_val,save_dir=save_name, loss_out=True)
                text_file = open(f"{save_name}/log.txt", "w")
                text_file.write(record_header + "\n")
                for line in record_lines:
                    text_file.write(line + "\n")
                # text_file.write(f"loss : {loss_avg}\n")
                text_file.write(f"loss profile : {loss_profile}\n")
        writer.export_scalars_to_json(f"{save_name}/all_scalars.json")
        writer.close()
        print('\t\tFinished in %.1fs'%(t2-t1))
        text_file.close()

    def plot_train_curve(self, save_name):
        if not os.path.isdir(save_name):
            os.mkdir(save_name)
        fig = plt.figure(figsize=(6, 6), facecolor="white")
        ax1 = fig.add_subplot(111)
        ax1.plot(self.train_loss_history,label='training')
        ax1.plot(self.val_loss_history,label='validation')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Error')
        ax1.legend()
        fig.show()
        fig.savefig(fname=f"{save_name}/train.png")
        fig.savefig(fname=f"{save_name}/train.pdf")
        plt.close()

    def save(self,name):
        data = {"tran_loss_hist":self.train_loss_history,"val_loss_hist":self.val_loss_history}
        torch.save(data,name+'_log.pt')
        torch.save(self.net,name+'_model.pt')

    def plot_test(self, test, loss_fn, save_dir, loss_out=False):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir) 
        test_size = test.data_size
        seg_angles = test.y_size
        total_loss = 0
        loss_list = []
        max_loss = float('-inf')
        min_loss = float('inf')
        idx_argmax = None
        idx_argmin = None
        for indx in range(test_size):
            test_x, test_y, test_z = test.get_batch(1, indx)
            self.net.eval()
            with torch.no_grad():
                predict_test = self.net(torch.as_tensor(test_x)).cpu().detach().numpy()
            pred_loss = loss_fn(torch.Tensor(test_y[0]).reshape((1, -1)), torch.Tensor(predict_test[0]).reshape((1, -1)))
            total_loss += pred_loss
            loss_list.append(pred_loss)
            if loss_out:
                if pred_loss > max_loss:
                    max_loss = pred_loss
                    idx_argmax= indx 
                if pred_loss < min_loss:
                    min_loss = pred_loss
                    idx_argmin= indx 
            xdata_show = test_x
            # print('xdata_show.shape: ', xdata_show.shape)
            rgbs = np.ones((N, 3))
            adjust_ratio = 0.85
            matrix_len=xdata_show.flatten().shape[0]
            # print('matrix_len: ', matrix_len)
            if matrix_len==6:
                num_panels=4
                matrix_shape = [2,3]
            else:
                num_panels=matrix_len
                dsize=int(np.sqrt(matrix_len))
                matrix_shape = [dsize, dsize]
            # print(num_panels, matrix_shape)
            xdata_show = xdata_show.reshape(matrix_shape[::-1])
            xdata_show = np.transpose(xdata_show)
            xdata_show = np.flip(xdata_show, 0)
            xdata_show = np.flip(xdata_show, 1)
            xx = max(colors_max)*(1-adjust_ratio)*((num_panels/matrix_len)==1)#70
            rgbs[:, 0] = np.linspace(((colors_max[0]-255)/N*xx+255)/255, colors_max[0]/255, N) # R
            rgbs[:, 1] = np.linspace(((colors_max[1]-255)/N*xx+255)/255, colors_max[1]/255, N) # G
            rgbs[:, 2] = np.linspace(((colors_max[2]-255)/N*xx+255)/255, colors_max[2]/255, N)  # B
            own_cmp = ListedColormap(rgbs)
            fig = plt.figure(figsize=(18, 6), facecolor='white')
            ax0 = fig.add_subplot(1, 3, 1)
            ax0.imshow(xdata_show, interpolation='nearest', cmap=own_cmp)
            ax0.axes.get_xaxis().set_visible(False)
            ax0.axes.get_yaxis().set_visible(False)
            ax1 = fig.add_subplot(1, 3, 2)
            # ax1.plot(np.linspace(-180, 180, seg_angles + 1)[0:seg_angles], test_y[0], label='Simulated', color=rgb_to_hex(real_rgb))
            # ax1.plot(np.linspace(-180, 180, seg_angles + 1)[0:seg_angles], predict_test[0], label='Predicted', color=rgb_to_hex(pred_rgb))
            ax1.plot(np.linspace(-180, 180, seg_angles + 1)[0:seg_angles], test_y[0][::-1], label='Simulated', color=rgb_to_hex(real_rgb))  #!
            ax1.plot(np.linspace(-180, 180, seg_angles + 1)[0:seg_angles], predict_test[0][::-1], label='Predicted', color=rgb_to_hex(pred_rgb))
            ax1.legend()
            ax1.set_xlabel('deg')
            ax1.set_xlim([-180, 180])
            ax1.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
            ax1.set_title(f'Dist (cm): {test_z[0, 0]} / Ang (deg): {test_z[0, 1]}\nLoss: {pred_loss}')
            fig.show()
            fig.savefig(fname=f"{save_dir}/test_{indx}.png")
            fig.savefig(fname=f"{save_dir}/test_{indx}.pdf")
            plt.close()

            ax = fig.add_subplot(1, 3, 3, polar=True)
            theta_rad = np.linspace(-180, 180, seg_angles + 1)[0:seg_angles] * np.pi/180
            # r_pred = predict_test[0][::-1]
            # r_real = test_y[0][::-1]
            r_pred = predict_test[0][::-1]
            r_real = test_y[0][::-1]
            ax.plot(theta_rad,r_real, drawstyle='steps', linestyle='-', color=rgb_to_hex(real_rgb), linewidth=2)  
            ax.plot(theta_rad, r_pred, drawstyle='steps', linestyle='-', color=rgb_to_hex(pred_rgb), linewidth=2)
            ax.set_yticklabels([])  # Hide radial tick labels
            ax.set_rticks(np.linspace(0, 1, 10))   
            ax.spines['polar'].set_visible(True)  # Show the radial axis line
            # Set the theta direction to clockwise
            ax.set_theta_direction(-1)
            # Set the theta zero location to the top
            ax.set_theta_zero_location('N')
            # ax.set_rlabel_position(-22.5)
            ax.set_theta_offset(np.pi / 2.0)
            ax.tick_params(axis='x', which='major', pad=14, labelsize=13)
            # ax.grid(True)
            fig.savefig(fname=f"{save_dir}/test_{indx}.png")
            fig.savefig(fname=f"{save_dir}/test_{indx}.pdf")
            plt.close()
            # ax2.set_frame_on(False)

        loss_avg = total_loss / test_size
        if loss_out:
            return {'avg': loss_avg, 'max': max_loss, 'argmax': idx_argmax, 'min': min_loss, 'argmin': idx_argmin}  #, 'list': loss_list}
        return {'avg': loss_avg}


from pyemd import emd
n=64
M1=np.zeros([n,n])
M2=np.zeros([n,n])
for i in range(n):
    for j in range(n):
        M1[i,j]=min(abs(i-j),j+n-i,i+n-j)#**2
        M2[i,j]=min(abs(i-j),j+n-i,i+n-j)**2

from utils.emd_sinkhorn_torch import emd_pop_zero_batch, sinkhorn_torch

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


# %%
