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

tetris_mode=True   # True if the detector is Tetris-inspired detector. False if it is a square detector
input_shape = 'J'  # [2, 5, 10, etc] (int) the size of the square detector. ['J', 'L', 'S', 'T', 'Z'] (string) for tetris detector.
seg_angles = 64 # segment of angles
file_header = '230121-192203_230121-161457_200_1' # save name of the model
recordpath = f'./save/mapping_data/{file_header}'
model_path = f'./save/models/{file_header}_model.pt'
model =torch.load(model_path)
RSID = np.array([[-4.0,11.0]]) # np.array([[1.0,2.0],[-3.0,14.0]])  #　The locations of radiation sources / an array with shape (n, 2)  (n: the number of radiation sources)
rot_ratio = 0 # rotation ratio X, where \phi = X\theta


DT = 0.1
SIM_TIME = 60.0
STATE_SIZE = 4
source_energies = [0.5e6 for _ in range(RSID.shape[0])]
SIM_STEP=10
num_particles = 50000
sim_parameters = {
    'DT': DT,
    'SIM_TIME': SIM_TIME,
    "STATE_SIZE": STATE_SIZE,
    'RSID':RSID,
    'source_energies':source_energies,
    'SIM_STEP':SIM_STEP,
    'rot_ratio': rot_ratio, 
    'num_particles': num_particles
}

# Map
map_horiz = [-15,15,30]
map_vert = [-5,25,30]

colors_parameters = {'array_hex':'#EEAD0E', 'pred_hex':'#CA6C4A' , 'real_hex': '#77C0D2'}

# recordpath = './save/mapping_data/' + file_header
if __name__ == '__main__' and record_data:
    if not os.path.isdir(recordpath):
        os.mkdir(recordpath)
    os.system('rm ' + recordpath + "/*")
jsonpath = recordpath + "_json/"
if __name__ == '__maintribution__' and record_data:
    if not os.path.isdir(jsonpath):
        os.mkdir(jsonpath)
    os.system('rm ' + jsonpath + "*")
figurepath = recordpath + "_figure/"
if __name__ == '__main__' and record_data:
    if not os.path.isdir(figurepath):
        os.mkdir(figurepath)
    os.system('rm ' + figurepath + "*")
predictpath = recordpath + "_predicted/"
if __name__ == '__main__' and record_data:
    if not os.path.isdir(predictpath):
        os.mkdir(predictpath)
    os.system('rm ' + predictpath + "*")


if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")

DEFAULT_DTYPE = torch.double

main(recordpath, tetris_mode, input_shape, seg_angles, model, sim_parameters, colors_parameters, device=DEFAULT_DEVICE)
write_data(seg_angles, recordpath, map_horiz, map_vert)

