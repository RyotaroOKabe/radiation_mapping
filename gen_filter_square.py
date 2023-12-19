# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import openmc
from utils.mcsimulation_square import *

#=========================set values here==========================
num_sources = 1 # the number of radiation sources to place.
a_num = 2  # The parameter for the configuration of the detector: a x a square shape
num_data = 64 # the number of the generated data
seg_angles = num_data # The number of angle sectors ( augnlar resolution: 360 deg/seg_angles)
source_energies = [0.5e6 for _ in range(num_sources)]    # Photon energy [eV]
header_dist_particles_dict = {'near': [50, 50000], 'far': [500, 50000]} 
run_name = 'sq2_1_filter'   # time.strftime('%y%m%d-%H%M%S', time.localtime()) # the folder name   #!
#=================================================================

openmc_dir = 'save/openmc_filter/'
save_fig = True
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
