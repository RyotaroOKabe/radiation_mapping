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
from utils.mcsimulation_square import *

num_sources = 1
a_num = 5   # The shape of the detector: a x a square
num_data = 64 # the number of the generated data
seg_angles = num_data # The number of angle sectors (resolution: 360 deg/seg_angles)
source_energies = [0.5e6 for l in range(num_sources)]    # Photon energy [eV]
header_dist_particles_dict = {'far': [500, 50000]}   # [distance (cm), the number of photon]
openmc_dir = 'save/openmc_filter/'
save_fig = True
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
folder=f'{openmc_dir}{run_name}'
angle_list = [0.1+a*360/num_data -180 for a in range(num_data)]
normalize=False
record = [f"run_name: {run_name}",
          f"folder: {folder}",
          f"num_sources: {num_sources}",
          f"a_num: {a_num}",
          f"num_data: {num_data}",
          f"seg_angles: {seg_angles}",
          f"source_energies: {source_energies}",
          f"header_dist_particles_dict: {header_dist_particles_dict}"]
print([r+"\n" for r in record])
print("angle_list for " + str(num_data) +" sections of angles:")
for header in header_dist_particles_dict.keys():
    dist = header_dist_particles_dict[header][0]
    num_particles = header_dist_particles_dict[header][1]
    for angle in angle_list:
        sources_d_th = [[dist, angle, source_energies[i]] for i in range(num_sources)]
        for i in range(num_sources):
            print(f"dist {i}: " + str(sources_d_th[i][0]))
            print(f"angle {i}: " + str(sources_d_th[i][1]))
            print(f"energy {i}: " + str(sources_d_th[i][2]))
        before_openmc(a_num, sources_d_th, num_particles)
        openmc.run()
        mm = after_openmc(a_num, sources_d_th, folder, seg_angles, header, normalize, record, savefig=save_fig)
