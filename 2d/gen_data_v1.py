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

folder1=f'openmc/data_{a_num}x{a_num}_{num_sources}src_{seg_angles}_data_20220815_v1.1/'
folder2=f'openmc/data_{a_num}x{a_num}_{num_sources}src_{seg_angles}_fig_20220815_v1.1/'

for i in range(num_data):
    sources_d_th = [[np.random.randint(dist_min, dist_max), 
                     float(np.random.randint(0, 360) + np.random.random(1)), 
                     source_energies[i]] for i in range(num_sources)]
    #?print("dist: " + str(rad_dist))
    for i in range(num_sources):
        print(f"dist {i}: " + str(sources_d_th[i][0]))
        print(f"angle {i}: " + str(sources_d_th[i][1]))
        print(f"energy {i}: " + str(sources_d_th[i][2]))
    before_openmc(a_num, sources_d_th, num_particles, seg_angles)
    run_openmc()
    mm = after_openmc(a_num, sources_d_th, folder1, folder2, seg_angles, header)
