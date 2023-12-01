# -*- coding: utf-8 -*-
#%%
import numpy as np
from os.path import join
# from pydrake.all import MathematicalProgram, Solve
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as matplotlib_polygon
from matplotlib.cm import get_cmap
import matplotlib
import math
# from pydrake.math import sin, cos, sqrt
import pickle as pkl
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
import os,sys
import random
random.seed(42)  
# import dill #!20220316
import imageio
from utils.mapping import mapping, gen_gif, Map
from utils.jv_data import *

th_level = 0.5  # Threshold level of the map
# file_idx=2   # TThe name of the data that is same as the one in 'run-detector.py'
fig_header = f'jvdata2_r3_v4.4'
recordpath = f'./save/mapping_data/{fig_header}'
fig_folder = f'./save/radiation_mapping/{fig_header}_{th_level}_v1.1'   #v1.2.1'
data_folder = './data/jayson/'
file_idx = 2
# map_horiz = [4,14,20]   #[0,10,20]
# map_vert = [-5,5,20]
acolors = ['r', 'g'] # arrow colors, [x+, y+] default=colors=['#77AE51', '#8851AE']):
#%%

def plot_3dprocess(data, step, res=100, cmap=get_cmap('Reds'), z_shift=1, 
                   ranges=None, space_folder=None, use_cbar=False, elevation=30, azimuth=30, 
                   pclouds=None, times=None, gtpos=[5.0, 0.0, -0.025]):
    m = data['m']
    X, Y = np.meshgrid(m.x_list, m.y_list)
    # Increase the resolution by interpolating data
    x_new = np.linspace(X.min(), X.max(), res)
    y_new = np.linspace(Y.min(), Y.max(), res)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    intensity_new = griddata((X.ravel(), Y.ravel()), m.intensity.ravel(), (X_new, Y_new), method='cubic')
    print('intensity_new: ', intensity_new.shape)
    # Create a 3D figure
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
    if ranges is not None:
        ax.set_xlim(ranges[:2])
        ax.set_ylim(ranges[2:4])
        ax.set_zlim(ranges[4:])
    else:
        ax.set_zlim([-0.5, 2])
    # Set the z-value for the 3D plot to 0
    Z = np.zeros_like(X_new)  # Assuming Z=0 for the entire plane
    hxtrue = data['hxTrue_data']
    px, py, pz = hxtrue[0, :], hxtrue[1, :], hxtrue[2, :]+ z_shift
    direction, angle = hxtrue[-2, -1], hxtrue[-1, -1]
    px1, py1, pz1 = px[-1], py[-1], pz[-1]

    mxwid, mywid = 3*(m.x_max - m.x_min), 3*(m.y_max - m.y_min)
    aratio = np.sqrt(mxwid**2+mywid**2)/np.sqrt(2*30**2)
    arrow_x0 = aratio*np.cos(direction)
    arrow_y0 = aratio*np.sin(direction)
    arrow_x1 = aratio*np.cos(angle)
    arrow_y1 = aratio*np.sin(angle)
    # Create a 3D surface plot with the higher-resolution data
    # surf = ax.plot_surface(X_new, Y_new, Z, facecolors=cmap(intensity_new), cmap=cmap, shade=False, zorder=0, alpha=None)
    # # Add a colorbar
    # if use_cbar:
    #     cbar = fig.colorbar(surf, ax=ax, label='Intensity', shrink=0.5)
    ax.plot(px, py, pz, c='k', lw=1.8, zorder=20)
    ax.plot(px, py, np.zeros_like(pz), c='#66CCCC', lw=1.8, zorder=10)
    horiz = np.linspace(0, pz1, 20)
    ax.plot(px1*np.ones_like(horiz), py1*np.ones_like(horiz), horiz, c='gray', lw=1.8, zorder=10, linestyle=':')
    ax.scatter(px1, py1, pz1, c='k', s=90, zorder=10)
    # ax.quiver(px1, py1, pz1, arrow_x0, arrow_y0, 0, length=1.0, arrow_length_ratio=0.3, color=acolors[0], zorder=30) # front side of the detector. 
    # ax.quiver(px1, py1, pz1, arrow_x1, arrow_y1, 0, length=1.0, arrow_length_ratio=0.3, color=acolors[1], zorder=40) # front side of the detector. 
    ax.quiver(px1, py1, np.zeros_like(pz1), arrow_x0, arrow_y0, 0, length=1.0, arrow_length_ratio=0.3, color=acolors[0], zorder=30) # front side of the detector. 
    ax.quiver(px1, py1, np.zeros_like(pz1), arrow_x1, arrow_y1, 0, length=1.0, arrow_length_ratio=0.3, color=acolors[1], zorder=40) # front side of the detector. 
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel('X (m)', fontsize=25, labelpad=17)
    ax.set_ylabel('Y (m)', fontsize=25, labelpad=20)
    ax.set_zlabel('Z (m)', fontsize=25, labelpad=15)
    ax.view_init(elev=elevation, azim=azimuth)
    if pclouds is not None:
        if 'z' in pclouds.keys():
            alpha_ = None
            zorder_ = 0
            x, y, z = pclouds['x'],pclouds['y'],pclouds['z']
            num_pc = x.shape[0]
            pc_indices = random.sample(range(num_pc), int(num_pc*0.01))
            ax.scatter(x[pc_indices], y[pc_indices], z[pc_indices], c='gray', marker='o', s=0.1, alpha=0.05, zorder=3)  # 'c' is color and 'marker' denotes the shape of the data point
        else:
            alpha_ = None
            zorder_ = 0
            x, y = pclouds['x'],pclouds['y']
            num_pc = x.shape[0]
            pc_indices = random.sample(range(num_pc), int(num_pc*0.01))
            ax.scatter(x[pc_indices], y[pc_indices], np.zeros_like(x[pc_indices])-0.1, c='gray', marker='o', s=0.1, alpha=0.05, zorder=3)  # 'c' is color and 'marker' denotes the shape of the data point
    else: 
        zorder_ = 0
        alpha_ = None
    surf = ax.plot_surface(X_new, Y_new, Z, facecolors=cmap(intensity_new), cmap=cmap, shade=False, zorder=zorder_, alpha=alpha_)
    if gtpos is not None: 
        ax.scatter(gtpos[0], gtpos[1], gtpos[2], marker ="x", c='k', s=160, zorder=4)
    # Add a colorbar
    if use_cbar:
        cbar = fig.colorbar(surf, ax=ax, label='Intensity', shrink=0.5)
    if times is None:
        plt.title('STEP: %.4d'%step, fontsize=30, y=1.05)
    else: 
        plt.title('STEP: %.4d'%step + '    TIME: {:.2f} s'.format(times[step]), fontsize=30, y=1.05)
    if space_folder is not None:
        plt.savefig(fname=join(space_folder, 'STEP%.4d'%step +".png"))
        # plt.savefig(fname=join(space_folder, 'STEP%.4d'%step +".pdf"))
    # Show the plot
    plt.show()

#%%
# load data
jvdict = load_jvdata(file_idx, data_folder)
energy, times, det_id, jvdata, px, \
    py, pz, qw, qx, qy, qz, x, y, z = [jvdict[k] for k in ['energy', 'timestamp', 'det_id', 
                                        'data', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'x', 'y', 'z']]

#%%
step = 84
# elevation, azimuth=90, -90
elevation, azimuth=30, -70
with open(os.path.join(fig_folder,f'data_STEP%.4d'%step+'.pkl'),'rb') as f:
    data=pkl.load(f, encoding="latin1")
plot_3dprocess(data, step, res=100, cmap=get_cmap('Reds'), space_folder=None, use_cbar=True, elevation=elevation, azimuth=azimuth)

ranges = [x.min(), x.max(), y.min(), y.max(), z.min(), z.max()]
with open(os.path.join(fig_folder,f'data_STEP%.4d'%step+'.pkl'),'rb') as f:
    data=pkl.load(f, encoding="latin1")
plot_3dprocess(data, step, res=100, cmap=get_cmap('Reds'), z_shift=0, ranges=ranges, space_folder=None, use_cbar=True, elevation=elevation, azimuth=azimuth, times=times)


#%%
pclouds = {'x': x, 'y': y, 'z': z}
# elevation, azimuth=90, -90
elevation, azimuth=40, -70
ranges = [x.min(), x.max(), y.min(), y.max(), z.min(), z.max()]
z_shift = 1 if ranges is None else 0
files=os.listdir(fig_folder)
files=[file for file in sorted(files) if file.endswith('pkl')]
space_dir = join(fig_folder, f'space1.4')
if not os.path.isdir(space_dir):
    os.mkdir(space_dir)
os.system('rm ' + space_dir + "/*")
for step, file in enumerate(files):
    
    with open(os.path.join(fig_folder,f'data_STEP%.4d'%step+'.pkl'),'rb') as f:
        data=pkl.load(f, encoding="latin1")
    plot_3dprocess(data, step, res=150, cmap=get_cmap('Reds'), 
                   z_shift=z_shift, ranges=ranges, space_folder=space_dir, 
                   use_cbar=True, elevation=elevation, azimuth=azimuth, pclouds=pclouds, times=times)
    
with imageio.get_writer(f'{space_dir}/mapping.gif', mode='I') as writer:
    for figurename in sorted(os.listdir(space_dir)):
        if figurename.endswith('png'):
            image = imageio.imread(space_dir + '/' + figurename)
            writer.append_data(image)
print( f'Finish making a gif: {space_dir}/mapping.gif')
#%%