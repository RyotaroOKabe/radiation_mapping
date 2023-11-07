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
from utils.mapping_drake import mapping, gen_gif

th_level = 0.3  # Threshold level of the map
file_idx=2   # TThe name of the data that is same as the one in 'run-detector..py'
fig_header = f'jvdata{file_idx}_v2'
recordpath = f'./save/mapping_data/{fig_header}'
fig_folder = f'./save/radiation_mapping/{fig_header}_{th_level}_v1'
map_horiz = [2,12,20]   # map geometry (horizontal) [m]
map_vert = [1,9,20] # map geometry (vertical) [m]

mapping(fig_folder, fig_header, recordpath, map_geometry = [map_horiz, map_vert], threshold=th_level)   #,
gen_gif(fig_folder) 

