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
from utils.move_detector import openmc_simulation, main

tetris_mode=False
if tetris_mode:
    from utils.mcsimulation_tetris import *

else:
    from utils.mcsimulation_square import *
    a_num =5
    num_sources = 2
    seg_angles = 64
    file_header = f"A20221024_{a_num}x{a_num}_{num_sources}src_{seg_angles}_v3.5"
    recordpath = f'mapping_data/mapping_{file_header}'
    model_path = f'save_model/model_openmc_{a_num}x{a_num}_{num_sources}src_{seg_angles}_ep2000_bs256_20220812_v1.1_model.pt'
    model =torch.load(model_path)
    num_panels=a_num**2
    matrix_shape = [a_num, a_num]

DT = 0.1  # time tick [s]
SIM_TIME = 70.0
STATE_SIZE = 4  #!20221023
RSID = np.array([[1.0,2.0,0.5e6]])#,[-3.0,14.0,0.5e6]])
source_energies = [0.5e6, 0.5e6]
SIM_STEP=10
rot_ratio = 0  #!20221023
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

colors_max = [238,173,14] #[243, 194, 92] #[255,193,37] #[238,118,33] #[255,97,3]  #colors_max = [255, 100, 0]
pred_rgb = [202, 108, 74] #[91,91,91] #[202, 108, 74]
real_rgb = [119, 192, 210] #[30,30,30] #[119, 192, 210]

recordpath = 'save/mapping_data/mapping_' + file_header
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
    DEFAULT_DEVICE = torch.device("cuda:1")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")

DEFAULT_DTYPE = torch.double

#%%
# colormap
from matplotlib.colors import ListedColormap
N = 256


main(seg_angles, model, recordpath, sim_parameters)
write_data(seg_angles, recordpath, map_horiz, map_vert)

