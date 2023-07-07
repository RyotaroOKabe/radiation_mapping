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
model_header = "S_230118-203413_230120-224603" #? save name of the model
model_path = f'./save/models/{model_header}_model.pt'   #?

data_name = '230706-182456' #'230706-133717' #'230124-214356'   #?
data_path = f'./save/openmc_data/{data_name}'   #?

save_dir = "./save/training"
save_name = f"{data_name}"    #?
save_header = f"{save_dir}/{save_name}_{input_shape}"   #?

model =torch.load(model_path)   #?
loss_fn = lambda y, y_pred: emd_loss_ring(y, y_pred, r=1).item()

#%%
# Load the test dataset
data_set=Dataset(seg_angles,output_fun=get_output,path=data_path)   #?
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
rmax = 40

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

sorted_AB = sorted(zip(ang_list, loss_list, acc_list), reverse=False)
sorted_ang, sorted_loss, sorted_acc = zip(*sorted_AB)
plt.plot(sorted_ang, sorted_loss)


fig = plt.figure(figsize=(6, 6), facecolor='white')
ax = fig.add_subplot(111, polar=True)
for j in range(num_dist):
    ax.plot([np.pi*a/180 for a in sorted_ang][j:][::num_dist], sorted_loss[j:][::num_dist], drawstyle='steps', linestyle='-', color=colors[j]) 
ax.set_yticklabels([])  # Hide radial tick labels
# Add the radial axis
ax.set_rticks(np.linspace(0, rmax, 10))  # Adjust the range and number of radial ticks as needed
ax.set_rlim([0, rmax])
ax.spines['polar'].set_visible(True)  # Show the radial axis line
# Set the theta direction to clockwise
ax.set_theta_direction(-1)
# Set the theta zero location to the top
ax.set_theta_zero_location('N')


#%%


