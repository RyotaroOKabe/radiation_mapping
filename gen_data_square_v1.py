#%%
"""
Created on 2022/12/27

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
from utils.mcsimulation_square import *

num_sources = 1
a_num = 2  # The shape of the detector: a x a square 
num_data = 1000 # the number of the generated data
seg_angles = 64 # The number of angle sectors (resolution: 360 deg/seg_angles)
dist_min = 20   # minimum distance between the radiation source and the detector (cm).
dist_max = 500 #500 # maximum distance between the radiation source and the detector (cm).

source_energies = [0.5e6 for _ in range(num_sources)]    # Photon energy [eV]
num_particles = 10000 #!50000   # The number of photon
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
header = 'data'
openmc_dir = 'save/openmc_data/'
save_fig = True
folder=f'{openmc_dir}{run_name}'

record = [f"run_name: {run_name}",
          f"folder: {folder}",
          f"num_sources: {num_sources}",
          f"a_num: {a_num}",
          f"num_data: {num_data}",
          f"seg_angles: {seg_angles}",
          f"Dist range: {[dist_min, dist_max]}",
          f"source_energies: {source_energies}",
          f"num_particles: {num_particles}"]
print([r+"\n" for r in record])

#%%
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
    before_openmc(a_num, sources_d_th, num_particles)
    openmc.run()
    mm = after_openmc(a_num, sources_d_th, folder, seg_angles, header, record, savefig=save_fig)
    json.dump(source_pos, open(f'{folder}/source_positions.json', 'w'))
    if save_fig:
        fig, ax = plt.subplots(1,1, figsize=(10, 10))
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
        fig.patch.set_facecolor('white')
        fig.savefig(f'{folder}_fig/source_positions.png')
        fig.savefig(f'{folder}_fig/source_positions.pdf')
        plt.close()
