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
from utils.dataset import get_output, FilterData2, load_data
from utils.emd_ring_torch import emd_loss_ring

tetris_mode=True    # True if the detector is Tetris-inspired detector. False if it is a square detector
input_shape = 'S'  # [2, 5, 10, etc] (int) the size of the square detector. ['J', 'L', 'S', 'T', 'Z'] (string) for tetris detector.
seg_angles = 64 # segment of angles
file_header = "S_230118-203413_230120-224603" # save name of the model
model_path = f'./save/models/{file_header}_model.pt'
recordpath = f'./save/mapping_data/{file_header}x'
data_name = '230124-214356' # '221227-001319'
data_path = f'./save/openmc_data/{data_name}'  #!20220716
model =torch.load(model_path)
loss_fn = lambda y, y_pred: emd_loss_ring(y, y_pred, r=1).item()

#%%
# Load the test dataset
train_set, test_set = load_data(test_size=100, train_size=None, test_size_gen=None, seg_angles=64,
                     output_fun=get_output, path=data_path, source_num=[1], prob=[1.0], seed=None)
#%%
# Perform inference on the test dataset
if not os.path.isdir(recordpath):
    os.mkdir(recordpath)
test_size = test_set.data_size
seg_angles = test_set.y_size
total_loss = 0
#%%

for indx in range(test_size):
    test_x,test_y,test_z=test_set.get_batch(1,indx)
    model.eval()
    with torch.no_grad():
        predict_test = model(torch.as_tensor(test_x)).cpu().detach().numpy()
    print(predict_test.shape)
    pred_loss = loss_fn(torch.Tensor(test_y[0]).reshape((1, -1)), torch.Tensor(predict_test[0]).reshape((1, -1)))
    total_loss += pred_loss
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
    fig.savefig(fname=f"{recordpath}/test_{indx}.png")
    fig.savefig(fname=f"{recordpath}/test_{indx}.pdf")

#%%