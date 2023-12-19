#%%
# -*- coding: utf-8 -*-
import numpy as np
import sys,os
sys.path.append('./')
from utils.model import * 
from utils.cal_param import *
from utils.move_detector import main
from utils.unet import *

#=========================set values here==========================
# Load models
input_shape = 2  # [2, 5, 10, etc] (int) the size of the square detector. Or ['J', 'L', 'S', 'T', 'Z'] (string) for tetris detector.
tetris_mode = True if isinstance(input_shape, str) else False   # True if the detector is Tetris-inspired detector. False if it is a square detector
seg_angles = 64 # The number of angle sectors ( augnlar resolution: 360 deg/seg_angles). Make sure to use the same value as those of training data and filters
model_name = '230124-214356_230121-165924_200_1_model.pt' # model name trained with the code 'train_model.py'
file_header = model_name.replace('_model.pt', '')  # The folder name header for saving files
model_path = './save/models/' + model_name
model =torch.load(model_path)

# Detector motion setting 
RSID = np.array([[-4.0,11.0]]) #  The position of radiation source(s). The shape of the array is (n, 2), where n is the number of radiation sources.
rot_ratio = 0 # rotation ratio \chi , where \phi = \chi * \theta
DT = 0.1    # digit of time (s)
SIM_TIME = 60.0 # The total simulation time (s)
STATE_SIZE = 4  # state size
source_energies = [0.5e6 for _ in range(RSID.shape[0])] # source energy (eV).
SIM_STEP=10 # simulation time step each of which we save pkl files.
num_particles = 50000   # The number of photon in MC simulation

# Specify area for radiation map
map_horiz = [-15,15,30] # map geometry (horizontal) [m]
map_vert = [-5,25,30] # map geometry (vertical) [m]

# Color setting 
colors_parameters = {'array_hex':'#EEAD0E', 'pred_hex':'#CA6C4A' , 'real_hex': '#77C0D2'}
#=================================================================

#%%
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

recordpath = f'./save/mapping_data/{file_header}'
jsonpath = recordpath + "_json/"
figurepath = recordpath + "_figure/"
predictpath = recordpath + "_predicted/"

if __name__ == '__main__' and record_data:
    if not os.path.isdir(recordpath):
        os.mkdir(recordpath)
    os.system('rm ' + recordpath + "/*")
    if not os.path.isdir(jsonpath):
        os.mkdir(jsonpath)
    os.system('rm ' + jsonpath + "*")
    if not os.path.isdir(figurepath):
        os.mkdir(figurepath)
    os.system('rm ' + figurepath + "*")
    if not os.path.isdir(predictpath):
        os.mkdir(predictpath)
    os.system('rm ' + predictpath + "*")


if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")

DEFAULT_DTYPE = torch.double

#%%
main(recordpath, tetris_mode, input_shape, seg_angles, model, sim_parameters, colors_parameters, device=DEFAULT_DEVICE)
write_data(seg_angles, recordpath, map_horiz, map_vert)

