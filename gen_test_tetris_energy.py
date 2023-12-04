# -*- coding: utf-8 -*-
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
from utils.mcsimulation_tetris import *

num_sources = 1
shape_name = 'S' # Tetris shape
num_data = 64 # the number of the generated data
seg_angles = num_data # The number of angle sectors (resolution: 360 deg/seg_angles)
# source_energies = [0.5e6 for l in range(num_sources)]    # Photon energy [eV]
e_coeff = 0.5
e_power = [4,5,6,7,8]
header_dist_particles_dict = {'b': [100, 20000]}    # [distance (cm), the number of photon]
openmc_dir = 'save/openmc_data/'
save_fig = True
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
folder=f'{openmc_dir}{run_name}'
use_panels = get_tetris_shape(shape_name)
# angle_list = [0.1+a*360/num_data -180 for a in range(num_data)]
angle_list = [0, 45, 90, 135, 180, 225, 270, 315]
angle_list = [0.1+a for a in angle_list]
record = [f"run_name: {run_name}",
          f"folder: {folder}",
          f"num_sources: {num_sources}",
          f"shape_name: {shape_name}",
          f"num_data: {num_data}",
          f"seg_angles: {seg_angles}",
          f"source_energies: {[e_coeff, e_power]}",
          f"header_dist_particles_dict: {header_dist_particles_dict}"]
print([r+"\n" for r in record])
print("angle_list for " + str(num_data) +" sections of angles:")
for header in header_dist_particles_dict.keys():
    for ep in e_power:
        source_energies = [e_coeff*10**ep for l in range(num_sources)] 
        header1 = f'{header}{e_coeff}e{ep}'
        dist = header_dist_particles_dict[header][0]
        num_particles = header_dist_particles_dict[header][1]
        for angle in angle_list:
            sources_d_th = [[dist, angle, source_energies[i]] for i in range(num_sources)]
            for i in range(num_sources):
                print(f"dist {i}: " + str(sources_d_th[i][0]))
                print(f"angle {i}: " + str(sources_d_th[i][1]))
                print(f"energy {i}: " + str(sources_d_th[i][2]))
            before_openmc(use_panels, sources_d_th, num_particles)
            openmc.run()
            mm = after_openmc(use_panels, sources_d_th, folder, seg_angles, header1, record, savefig=save_fig)
