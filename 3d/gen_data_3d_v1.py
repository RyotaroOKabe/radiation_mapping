#%%

"""

ROkabe
last update: 2022/07/05
original: gen_openmc_data_discrete_a3_v4.py

Major update: theta digit: 180 deg = 18*10 deg >>> 20*9 deg

"""

import numpy as np
#import torch   #!20220125 20220430 remove for its incompatibility
from datetime import datetime   #!20220224
import openmc
from mcsimulation_3d_v1 import *

GPU_INDEX = 0
USE_CPU = False


#%%
if __name__ == '__main__':
    a_num = 5   # The size of detector:  [a x a x a] detectors
    num_data = 3500
    dist = 50
    num_particles =100000
    dist_min = 100
    dist_max = 100
    ph_num = 32
    th_num = 16
    folder1=f'openmc/discrete_data_20220829_{a_num}^3_v1/'
    folder2=f'openmc/discrete_fig_20220829_{a_num}^3_v1/'
    header = "data"

    for i in range(num_data):
        rad_phi=float(np.random.randint(0, 360) + np.random.random(1))
        rad_th=float(np.random.randint(0, 180) + np.random.random(1))
        print("dist: " + str(dist))
        print("angle: " + str(rad_phi))
        before_openmc(a_num, dist, rad_phi, rad_th, ph_num, th_num, num_particles)  #!20220628 #!a_num
        openmc.run()
        mm = after_openmc(a_num, dist, rad_phi, rad_th, ph_num, th_num, folder1, folder2, header) #!20220706

# %%
