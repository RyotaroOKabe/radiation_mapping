#%%
# -*- coding: utf-8 -*-
import numpy as np
from pydrake.all import MathematicalProgram, Solve
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as matplotlib_polygon
import matplotlib
import math
from pydrake.math import sin, cos, sqrt
import pickle as pkl
from scipy.interpolate import interp2d
import os,sys
import dill #!20220316
import imageio
from utils.mapping import mapping, gen_gif

# factor1 = 1e+24
# a_num = 2
# num_sources = 1
# seg_angles = 64
# fig_folder = f'mapping_data/save_fig/'
# fig_header = f'A20221024_{a_num}x{a_num}_{num_sources}src_{seg_angles}_v3.5.1'
# record_path = f'mapping_data/mapping_A20221024_{a_num}x{a_num}_{num_sources}src_{seg_angles}_v3.5'   #'mapping_data/mapping_A20220804_10x10_v1.7'
fig_header = f"221227-001319"
recordpath = f'./save/mapping_data/{fig_header}'
fig_folder = f'./save/radiation_mapping/{fig_header}'
th_level = 0.2
map_horiz = [-15,15,30]
map_vert = [-5,25,30]

#%%
mapping(fig_folder, fig_header, recordpath, save_process=True, savedata=True)
# test()
gen_gif(fig_folder)

# %%
