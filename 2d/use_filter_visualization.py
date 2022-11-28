#%%
from contextlib import redirect_stderr
import glob
import imp
from IPython.display import Image
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scipy.stats
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import json
import time, timeit
from datetime import datetime
import openmc
from mcsimulation_square import *

#%%

filter_dir = 'openmc/disc_filter_2x2_64_data_20220812_v1.2'
#far_500_-5.525.json

onlyfiles = [f for f in listdir(filter_dir) if isfile(join(filter_dir, f))]

headers = ['near', 'far']
dist = [50, 500]
num_data = 64
a_num = 2

N = 256
blue = np.ones((N, 3))
blue[:, 0] = np.linspace(200/256, 0, N) # R = 255
blue[:, 1] = np.linspace(200/256, 0, N) # G = 232
blue[:, 2] = np.linspace(256/256, 139/256, N)  # B = 11
blue_cmp = ListedColormap(blue)

orange = np.ones((N, 3))
orange[:, 0] = np.linspace(256/256, 255/256, N) # R = 255
orange[:, 1] = np.linspace(240/256, 100/256, N) # G = 232
orange[:, 2] = np.linspace(200/256, 0/256, N)  # B = 11
orange_cmp = ListedColormap(orange)

orange = np.ones((N, 3))
orange_full = [238,173,14] #[255, 100, 0]
xx = 30#70
orange[:, 0] = np.linspace(((orange_full[0]-256)/N*xx+256)/256, orange_full[0]/256, N) # R = 255
orange[:, 1] = np.linspace(((orange_full[1]-256)/N*xx+256)/256, orange_full[1]/256, N) # G = 232
orange[:, 2] = np.linspace(((orange_full[2]-256)/N*xx+256)/256, orange_full[2]/256, N)  # B = 11
orange_cmp = ListedColormap(orange)


colors = orange_cmp #'Blues'
name = 'yellow'


angle_list = [0.1+a*360/num_data -180 for a in range(num_data)]

choice = 0
fig, axs = plt.subplots(8,8,figsize=(40,40))
# axs[0, 0].plot(x, y)
# axs[0, 0].set_title('Axis [0, 0]')

count = 0
for ang in angle_list:
    file_path = f'{filter_dir}/{headers[choice]}_{dist[choice]}_{round(ang, 3)}.json'
    print(file_path)
    data = json.load(open(file_path, 'rb'))
    detection = np.array(data['input']).reshape((a_num, a_num))
    detection = np.flip(np.transpose(detection), 0)

    # ds_ag_list = file2[:-5].split('_')[1:]  #!20220517
    # ds_ag_title = ''
    # for i in range(num_sources):
    #     ds, ag = ds_ag_list[2*i], ds_ag_list[2*i+1]
    #     ds_ag_line = f'dist{i}: {ds},  angle{i}: {ag}'
    #     if i != num_sources-1:
    #         ds_ag_line += '\n'
    #     ds_ag_title += ds_ag_line
    fig2, axs2 = plt.subplots(1,1,figsize=(6,6))
    plt.imshow(detection, interpolation='nearest', cmap=colors)#"plasma")
    # plt.set_title(ang)
    # axs2.set_xlabel('y')
    # axs2.set_ylabel('x')
    plt.colorbar()
    # plt.savefig(folder2 + file2)
    # plt.savefig(folder2 + file2[:-3] + 'pdf')
    # plt.show()
    plt.savefig(f'figures/filters/{headers[choice]}_{dist[choice]}_{round(ang, 3)}_{name}.png')
    plt.savefig(f'figures/filters/{headers[choice]}_{dist[choice]}_{round(ang, 3)}_{name}.pdf')
    # plt.close()
    # axs[count%8, count//8].imshow(detection, interpolation='nearest')
    # axs[count%8, count//8].set_title(round(ang, 3))
    axs[count//8, count%8].imshow(detection, interpolation='nearest', cmap=colors)
    # axs[count//8, count%8].set_title(round(ang, 3))
    axs[count//8, count%8].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    # print('json dir')
    # print(folder1+file1)
    # print('fig dir')
    # print(folder2+file2)
    # return mean
    count += 1

fig.patch.set_facecolor('white')
fig.savefig(f'figures/filters/{headers[choice]}_{name}_multiple.png')
fig.savefig(f'figures/filters/{headers[choice]}_{name}_multiple.pdf')

# %%
