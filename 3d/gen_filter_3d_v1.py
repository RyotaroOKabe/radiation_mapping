#%%

"""

ROkabe
last update: 2022/07/05
original: gen_openmc_data_discrete_a3_v4.py

Major update: theta digit: 180 deg = 18*10 deg >>> 20*9 deg

"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json #!20220119
#import torch   #!20220125 20220430 remove for its incompatibility
import timeit   #!20220224
from datetime import datetime   #!20220224
import openmc
from mcsimulation_3d_v1 import *

GPU_INDEX = 0
USE_CPU = False


#%%
if __name__ == '__main__':
    a_num = 5
    ph_num = 32
    th_num = 16
    folder1=f'openmc/disc_filter_data_20220829_{a_num}^3_v1/'
    folder2=f'openmc/disc_filter_fig_20220829_{a_num}^3_v1/'
    header_dist_particles_dict = {'near': [20, 50000], 'far': [100, 150000]}
    phi_list = [0.1+a*360/ph_num -180 for a in range(ph_num)]
    theta_list = [90/th_num+b*180/th_num for b in range(th_num)]

    for header in header_dist_particles_dict.keys():
        for i in range(ph_num):
            for j in range(th_num):
                dist = header_dist_particles_dict[header][0]
                num_particles = header_dist_particles_dict[header][1]
                rad_phi = phi_list[i]
                rad_th = theta_list[j]
                print("dist: " + str(dist))
                print("phi: " + str(rad_phi))
                print("theta: " + str(rad_th))
                before_openmc(a_num=a_num, rad_dist=dist, rad_phi=rad_phi, rad_th=rad_th, ph_num=ph_num, th_num=th_num, num_particles=num_particles)  #!20220628 #!a_num
                openmc.run()
                mm = after_openmc(a_num, dist, rad_phi, rad_th, ph_num, th_num, folder1, folder2, header) #!20220706

# %%
