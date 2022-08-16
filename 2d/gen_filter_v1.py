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
a_num = 2
num_data = 10
seg_angles = 64
dist_min = 50
dist_max = 500
source_energies = [0.5e6, 0.5e6]
num_particles = 20000
header = 'data'

num_sources = 1
a_num = 3
num_data = 64
seg_angles = num_data
dist_min = 50
dist_max = 500
source_energies = [0.5e6, 0.5e6]
header_dist_particles_dict = {'near': [50, 90000], 'far': [500, 90000]}
folder1=f'openmc/filter_{a_num}x{a_num}_{seg_angles}_data_20220815_v1.1/'
folder2=f'openmc/filter_{a_num}x{a_num}_{seg_angles}_fig_20220815_v1.1/'
angle_list = [0.1+a*360/num_data -180 for a in range(num_data)]
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
        before_openmc(a_num, sources_d_th, num_particles, seg_angles)
        openmc.run()
        mm = after_openmc(a_num, sources_d_th, folder1, folder2, seg_angles, header)
