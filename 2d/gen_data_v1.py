#%%
"""
Created on 2022/08/15

@author: R.Okabe
"""

from contextlib import redirect_stderr
import glob
import imp
from IPython.display import Image
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scipy.stats
import numpy as np
import pandas as pd
import os
import json
import time, timeit
from datetime import datetime
import openmc
from mcsimulation_square import *

num_sources = 1
a_num = 2   # The shape of the detector: a x a square 
num_data = 1000 # the number of the generated data
seg_angles = 64 # The number of angle sectors (resolution: 360 deg/seg_angles)
dist_min = 50   # minimum distance between the radiation source and the detector (cm).
dist_max = 1000 #500 # maximum distance between the radiation source and the detector (cm).
source_energies = [0.5e6 for l in range(num_sources)]    # Photon energy [eV]
num_particles = 20000 #!20000   # The number of photon
header = 'data'
openmc_dir = 'openmc/'
folder1=f'{openmc_dir}sq{a_num}_{num_sources}s_d{dist_min}to{dist_max}_a{seg_angles}_dat_221207_v1.1/'
folder2=f'{openmc_dir}sq{a_num}_{num_sources}s_d{dist_min}to{dist_max}_a{seg_angles}_fig_221207_v1.1/'

#%%
# if os.path.exists(f'{folder1}/source_positions.json'):
#     source_pos = json.load(open(f'{folder1}/source_positions.json', 'rb'))
#     # source_pos = json.load(f'{folder1}/source_positions.json')
# else:
source_pos = [] # N * (x,y)


for i in range(num_data):
    sources_d_th = [[np.random.randint(dist_min, dist_max), 
                     float(np.random.randint(0, 360) + np.random.random(1)), 
                     source_energies[ii]] for ii in range(num_sources)]
    #?print("dist: " + str(rad_dist))
    for j in range(num_sources):
        distance = sources_d_th[j][0] # [cm]
        angle = sources_d_th[j][1] # [deg]
        # x_pos = distance * np.cos(angle*np.pi/180)
        # y_pos = distance * np.sin(angle*np.pi/180)
        x_pos = distance * np.sin(angle*np.pi/180)
        y_pos = distance * np.cos(angle*np.pi/180)
        print(f"dist {i}/{j}: " ,distance)
        print(f"angle {i}/{j}: ", angle)
        print(f"energy {i}/{j}: ", sources_d_th[j][2])
        source_pos.append([x_pos,y_pos])
    before_openmc(a_num, sources_d_th, num_particles, seg_angles)
    run_openmc()
    mm = after_openmc(a_num, sources_d_th, folder1, folder2, seg_angles, header)
    json.dump(source_pos, open(f'{folder1}/source_positions.json', 'w'))
    # fig, axs = plt.subplots(1,2, figsize=(20, 10))
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    # ax = axs[0]
    ax.scatter(np.array(source_pos)[:,0], np.array(source_pos)[:,1], s=15, color='#64ADB1')
    ax.scatter(0,0, marker="s", color='k', s=30)  #!20220804 multi sources
    circle_max = plt.Circle((0, 0), dist_max, color='k', lw=1, fill=False)
    circle_min = plt.Circle((0, 0), dist_min, color='k', lw=1, fill=False)
    ax.add_patch(circle_max)
    ax.add_patch(circle_min)
    ax.set_xlim(-dist_max*1.1, dist_max*1.1)
    ax.set_ylim(-dist_max*1.1, dist_max*1.1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title(f'{a_num}x{a_num}_{num_sources}src_{seg_angles}/ Points: {i+1}', fontsize=15)
    #ax2 = axs[1]
    #ax2.hist([np.array(source_pos)[m,0]**2+np.array(source_pos)[m,1]**2 for m in range(len(source_pos))], range=(-10, 0), color='#64ADB1') #, bins=num_data//50
    fig.patch.set_facecolor('white')
    fig.savefig(f'{folder2}/source_positions.png')
    fig.savefig(f'{folder2}/source_positions.pdf')
