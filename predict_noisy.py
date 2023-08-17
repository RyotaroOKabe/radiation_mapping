#%%
# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import sys,os
import pickle as pkl
sys.path.append('./')   #!20220331
from utils.model import * 
import matplotlib.pyplot as plt #!20220509
from matplotlib.figure import Figure   
from matplotlib.patches import Wedge
import imageio  #!20220520
import openmc
from utils.cal_param import *   #!20221023
from utils.move_detector import main
from utils.unet import *
from utils.dataset import get_output, FilterData2, load_data, compute_accuracy, Dataset, Testset
from utils.emd_ring_torch import emd_loss_ring
colors = ['r', 'b', 'g', 'y']
plot_test = False
tetris_mode=True   # True if the detector is Tetris-inspired detector. False if it is a square detector
input_shape = 'S'  #? [2, 5, 10, etc] (int) the size of the square detector. ['J', 'L', 'S', 'T', 'Z'] (string) for tetris detector.
seg_angles = 64 # segment of angles
model_header = "230118-203413_230120-224603_200_1" #? save name of the model
model_path = f'./save/models/{model_header}_model.pt'   #?

data_name = '230706-182456' #'230706-133717' #'230124-214356'   #?
data_path = f'./save/openmc_data/{data_name}'   #?

save_dir = "./save/training"
save_name = f"{data_name}_kk"    #?
save_header = f"{save_dir}/{save_name}_{input_shape}"   #?

model =torch.load(model_path)   #?
loss_fn = lambda y, y_pred: emd_loss_ring(y, y_pred, r=1).item()

#%%

class SystemData:
    def __init__(self, model_header, data_name):
        self.model_header = model_header
        self.model_path = f'./save/models/{model_header}_model.pt'
        
        self.data_name = data_name
        self.data_path = f'./save/openmc_data/{data_name}'
        
        self.save_dir = "./save/training"
        self.save_name = f"{data_name}"
        self.save_header = f"{save_dir}/{save_name}"
        
        self.model = torch.load(self.model_path)

#%%

class NoisyDataset(object):
    """docstring for Datasedt"""
    def __init__(self, seg_angles, output_fun,path, noiselevel=0.1):
        super(NoisyDataset, self).__init__()
        files=os.listdir(path)
        self.names=[]
        self.source_list=[]
        xdata=[]
        ydata=[]
        zdata=[]
        for filename in files:
            if not filename.endswith('.json'):continue
            if filename.startswith('source'):continue
            with open(os.path.join(path,filename),'r') as f:
                data=json.load(f)
                self.names.append(filename)
                xdata.append(data['input'])
                ydata.append(output_fun(data['source'], seg_angles))
                zdata.append([float(da) for da in filename[:-5].split("_")[1:]])
                self.source_list.append(data['source'])
      
        xdata=np.array(xdata)
        ydata=np.array(ydata)
        zdata=np.array(zdata)

        self.std_dev = noiselevel  
        xsize = xdata.shape
        noise = np.random.normal(0, self.std_dev, xsize)
        xdata=xdata+noise

        self.xdata=xdata
        self.ydata=ydata
        self.zdata=zdata
        self.data_size=xdata.shape[0]

def calculate_mean(lst):
    total = sum(lst)
    mean = total / len(lst)
    return mean

#%%
# Load the test dataset
data_set=NoisyDataset(seg_angles,output_fun=get_output,path=data_path, noiselevel=0.1)   #?
test_set=Testset(data_set.xdata,data_set.ydata,data_set.zdata,test_size=None)   #?

#%%
# Perform inference on the test dataset
if not os.path.isdir(save_header):os.mkdir(save_header) #?
test_size = test_set.data_size  #?
seg_angles = test_set.y_size    #?
total_loss = 0  #?


#%%
loss_list = []  #?
acc_list = []   #?
ang_list = []   #?
num_dist = test_set.data_size//seg_angles   #?

for indx in range(test_size):   #?
    test_x,test_y,test_z=test_set.get_batch(1,indx)
    model.eval()
    with torch.no_grad():
        predict_test = model(torch.as_tensor(test_x)).cpu().detach().numpy()
    pred_loss = loss_fn(torch.Tensor(test_y[0]).reshape((1, -1)), torch.Tensor(predict_test[0]).reshape((1, -1)))
    total_loss += pred_loss
    loss_list.append(pred_loss)
    ang_list.append(test_z[0][-1])
    acc_list.append(compute_accuracy(torch.Tensor(test_y[0]).reshape(-1), torch.Tensor(predict_test[0])))
    if plot_test:
        fig = plt.figure(figsize=(6, 6), facecolor='white')
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(np.linspace(-180,180,seg_angles+1)[0:seg_angles],test_y[0],label='Simulated')
        ax1.plot(np.linspace(-180,180,seg_angles+1)[0:seg_angles],predict_test[0],label='Predicted')
        ax1.legend()
        ax1.set_xlabel('deg')
        ax1.set_xlim([-180,180])
        ax1.set_xticks([-180,-135,-90,-45,0,45,90,135,180])
        ax1.set_title(f'Dist (cm): {test_z[0,0]} / Ang (deg): {test_z[0,1]}\nLoss: {pred_loss}')
        fig.show()
        fig.savefig(fname=f"{save_header}/test_{indx}.png")
        fig.savefig(fname=f"{save_header}/test_{indx}.pdf")



#%%
loss_mean_list =[]
loss_list = []  #?
acc_list = []   #?
ang_list = []   #?
num_dist = test_set.data_size//seg_angles   #?
s_start=0.1
s_end = 2.0
s_num = 20
for sigma in np.linspace(s_start, s_end, s_num):
    data_set=NoisyDataset(seg_angles,output_fun=get_output,path=data_path, noiselevel=sigma)   #?
    test_set=Testset(data_set.xdata,data_set.ydata,data_set.zdata,test_size=None)   #?
    for indx in range(test_size):   #?
        test_x,test_y,test_z=test_set.get_batch(1,indx)
        model.eval()
        with torch.no_grad():
            predict_test = model(torch.as_tensor(test_x)).cpu().detach().numpy()
        pred_loss = loss_fn(torch.Tensor(test_y[0]).reshape((1, -1)), torch.Tensor(predict_test[0]).reshape((1, -1)))
        total_loss += pred_loss
        loss_list.append(pred_loss)
        ang_list.append(test_z[0][-1])
        acc_list.append(compute_accuracy(torch.Tensor(test_y[0]).reshape(-1), torch.Tensor(predict_test[0])))
        if plot_test:
            fig = plt.figure(figsize=(6, 6), facecolor='white')
            ax1 = fig.add_subplot(1,1,1)
            ax1.plot(np.linspace(-180,180,seg_angles+1)[0:seg_angles],test_y[0],label='Simulated')
            ax1.plot(np.linspace(-180,180,seg_angles+1)[0:seg_angles],predict_test[0],label='Predicted')
            ax1.legend()
            ax1.set_xlabel('deg')
            ax1.set_xlim([-180,180])
            ax1.set_xticks([-180,-135,-90,-45,0,45,90,135,180])
            ax1.set_title(f'Dist (cm): {test_z[0,0]} / Ang (deg): {test_z[0,1]}\nLoss: {pred_loss}')
            fig.show()
            fig.savefig(fname=f"{save_header}/test_{indx}.png")
            fig.savefig(fname=f"{save_header}/test_{indx}.pdf")
    print(sigma, calculate_mean(loss_list))
    loss_mean_list.append(calculate_mean(loss_list))

plt.plot(np.linspace(s_start, s_end, s_num), loss_mean_list, marker='o')


#%%
square = SystemData(model_header="230118-005847_230120-214857_200_1", data_name="230707-134940")
s_shape =SystemData(model_header="230118-203413_230120-224603_200_1", data_name="230706-210300")
j_shape =SystemData(model_header="230121-192203_230121-161457_200_1", data_name="230707-084745")
t_shape =SystemData(model_header="230124-214356_230121-165924_200_1", data_name="230707-111712")
classes = {'2x2 square': square, 'S-shape': s_shape, 'J-shape': j_shape, 'T-shape': t_shape}
colors = {'2x2 square': '#C76D38', 'S-shape': '#4EB15B', 'J-shape': '#3892C7', 'T-shape': '#B14EA4'}
figname = 'noisy_4panels'
ls_start=-2
ls_end = 0.2
ls_num = 12
fig = plt.figure(figsize=(7, 5), facecolor='white')
ax = fig.add_subplot(1,1,1)
for k, cl in classes.items(): 
    model = cl.model
    data_path = cl.data_path
    loss_mean_list =[]
    loss_list = [] 
    acc_list = []  
    ang_list = []  
    num_dist = test_set.data_size//seg_angles 

    for log_sigma in np.linspace(ls_start, ls_end, ls_num):
        sigma= np.power(10, log_sigma)
        data_set=NoisyDataset(seg_angles,output_fun=get_output,path=data_path, noiselevel=sigma)   #?
        test_set=Testset(data_set.xdata,data_set.ydata,data_set.zdata,test_size=None)   #?
        for indx in range(test_size):   #?
            test_x,test_y,test_z=test_set.get_batch(1,indx)
            model.eval()
            with torch.no_grad():
                predict_test = model(torch.as_tensor(test_x)).cpu().detach().numpy()
            pred_loss = loss_fn(torch.Tensor(test_y[0]).reshape((1, -1)), torch.Tensor(predict_test[0]).reshape((1, -1)))
            total_loss += pred_loss
            loss_list.append(pred_loss)
            ang_list.append(test_z[0][-1])
            acc_list.append(compute_accuracy(torch.Tensor(test_y[0]).reshape(-1), torch.Tensor(predict_test[0])))
            if plot_test:
                fig0 = plt.figure(figsize=(6, 6), facecolor='white')
                ax1 = fig0.add_subplot(1,1,1)
                ax1.plot(np.linspace(-180,180,seg_angles+1)[0:seg_angles],test_y[0],label='Simulated')
                ax1.plot(np.linspace(-180,180,seg_angles+1)[0:seg_angles],predict_test[0],label='Predicted')
                ax1.legend()
                ax1.set_xlabel('deg')
                ax1.set_xlim([-180,180])
                ax1.set_xticks([-180,-135,-90,-45,0,45,90,135,180])
                ax1.set_title(f'Dist (cm): {test_z[0,0]} / Ang (deg): {test_z[0,1]}\nLoss: {pred_loss}')
                fig0.show()
                fig0.savefig(fname=f"{save_header}/test_{indx}.png")
                fig0.savefig(fname=f"{save_header}/test_{indx}.pdf")
        print(sigma, calculate_mean(loss_list))
        loss_mean_list.append(calculate_mean(loss_list))

    ax.plot(np.linspace(ls_start, ls_end, ls_num), loss_mean_list, marker='o', label=k, color=colors[k], lw=2)

ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('$log(\sigma)$', fontsize=16)  # X-axis label fontsize
ax.set_ylabel('Wasserstein Loss', fontsize=16) 
ax.legend(frameon=False, fontsize=15)
fig.savefig(f'./figures/{figname}.png')
fig.savefig(f'./figures/{figname}.pdf')

#%%


