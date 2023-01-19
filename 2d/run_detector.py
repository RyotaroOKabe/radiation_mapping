#%%

"""
2022/08/15
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys,os
import pickle as pkl

#sys.path.append('../')
sys.path.append('./')   #!20220331
from utils.model import *  #!20220717
#?from train_torch_openmc_tetris_v1 import *  #!20220717

import matplotlib.pyplot as plt #!20220509
from matplotlib.figure import Figure   
from matplotlib.patches import Wedge
import imageio  #!20220520
import openmc
# from mcsimulation_tetris import *
from utils.cal_param import *   #!20221023
from utils.move_detector import main
from utils.unet import *

tetris_mode=False
# if tetris_mode:
#     from utils.mcsimulation_tetris import *

# else:
#     from utils.mcsimulation_square import *
a_num =10
num_sources = 1
seg_angles = 64
# file_header = "230103-003600"    #f"221227-001319"    #!
file_header = "{data_name}_{filter_name}"  # save_name = f"{data_name}" #!
recordpath = f'./save/mapping_data/{file_header}'
model_path = f'./save/models/{file_header}_model.pt'
model =torch.load(model_path)
num_panels=a_num**2
matrix_shape = [a_num, a_num]

DT = 0.1  # time tick [s]
SIM_TIME = 70.0
STATE_SIZE = 4
RSID = np.array([[1.0,2.0,0.5e6],[-3.0,14.0,0.5e6]]) #np.array([[1.0,2.0,0.5e6]])  #,[-3.0,14.0,0.5e6]])
source_energies = [0.5e6 for _ in range(RSID.shape[0])]
SIM_STEP=10
rot_ratio = 0
sim_parameters = {
    'DT': DT,
    'SIM_TIME': SIM_TIME,
    "STATE_SIZE": STATE_SIZE,
    'RSID':RSID,
    'source_energies':source_energies,
    'SIM_STEP':SIM_STEP,
    'rot_ratio': rot_ratio
}

# Map
map_horiz = [-15,15,30]
map_vert = [-5,25,30]

colors_parameters = {'array_hex':'#EEAD0E', 'pred_hex':'#CA6C4A' , 'real_hex': '#77C0D2'}

recordpath = './save/mapping_data/' + file_header
if __name__ == '__main__' and record_data:
    if not os.path.isdir(recordpath):
        os.mkdir(recordpath)
    os.system('rm ' + recordpath + "/*")    #!20220509
jsonpath = recordpath + "_json/"
if __name__ == '__maintribution__' and record_data:
    if not os.path.isdir(jsonpath):
        os.mkdir(jsonpath)
    os.system('rm ' + jsonpath + "*")    #!20220509
figurepath = recordpath + "_figure/"
if __name__ == '__main__' and record_data:
    if not os.path.isdir(figurepath):
        os.mkdir(figurepath)
    os.system('rm ' + figurepath + "*")    #!20220509
predictpath = recordpath + "_predicted/"
if __name__ == '__main__' and record_data:
    if not os.path.isdir(predictpath):
        os.mkdir(predictpath)
    os.system('rm ' + predictpath + "*")    #!20220509


if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")

DEFAULT_DTYPE = torch.double

#%%
# colormap
from matplotlib.colors import ListedColormap
# N = 256

# main(recordpath, tetris_mode, a_num, seg_angles, model, sim_parameters, device=DEFAULT_DEVICE)
main(recordpath, tetris_mode, a_num, seg_angles, model, sim_parameters, colors_parameters, device=DEFAULT_DEVICE)
write_data(seg_angles, recordpath, map_horiz, map_vert)

