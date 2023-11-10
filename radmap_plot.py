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
# import dill #!20220316
import imageio
from utils.mapping import mapping, gen_gif, Map

th_level = 0.5  # Threshold level of the map
# file_idx=2   # TThe name of the data that is same as the one in 'run-detector.py'
fig_header = f'jvdata2_r3_v3.1'
recordpath = f'./save/mapping_data/{fig_header}'
fig_folder = f'./save/radiation_mapping/{fig_header}_{th_level}_v1.2'
# map_horiz = [4,14,20]   #[0,10,20]
# map_vert = [-5,5,20]
acolors = ['r', 'g'] # arrow colors, [x+, y+] default=colors=['#77AE51', '#8851AE']):
#%%

def plot_3dprocess(data, step, res=100, cmap=get_cmap('Reds'), use_cbar=False):
    m = data['m']
    X, Y = np.meshgrid(m.x_list, m.y_list)
    # Increase the resolution by interpolating data
    x_new = np.linspace(X.min(), X.max(), res)
    y_new = np.linspace(Y.min(), Y.max(), res)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    intensity_new = griddata((X.ravel(), Y.ravel()), m.intensity.ravel(), (X_new, Y_new), method='cubic')

    # Create a 3D figure
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
    ax.set_zlim([-0.5, 2])
    # Set the z-value for the 3D plot to 0
    Z = np.zeros_like(X_new)  # Assuming Z=0 for the entire plane
    hxtrue = data['hxTrue_data']
    px, py, pz = hxtrue[0, :], hxtrue[1, :], hxtrue[2, :]+1
    direction, angle = hxtrue[-2, -1], hxtrue[-1, -1]
    px1, py1, pz1 = px[-1], py[-1], pz[-1]

    mxwid, mywid = 3*(m.x_max - m.x_min), 3*(m.y_max - m.y_min)
    aratio = np.sqrt(mxwid**2+mywid**2)/np.sqrt(2*30**2)
    arrow_x0 = aratio*np.cos(direction)
    arrow_y0 = aratio*np.sin(direction)
    arrow_x1 = aratio*np.cos(angle)
    arrow_y1 = aratio*np.sin(angle)
    # Create a 3D surface plot with the higher-resolution data
    surf = ax.plot_surface(X_new, Y_new, Z, facecolors=cmap(intensity_new), cmap=cmap, shade=False, zorder=0, alpha=None)
    # Add a colorbar
    if use_cbar:
        cbar = fig.colorbar(surf, ax=ax, label='Intensity', shrink=0.5)
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
    ax.set_xlabel('X', fontsize=25, labelpad=15)
    ax.set_ylabel('Y', fontsize=25, labelpad=15)
    ax.set_zlabel('Z', fontsize=25, labelpad=10)
    plt.title('STEP:  %.4d'%step, fontsize=30, y=1.05)
    plt.savefig(fname=fig_folder+'/'+ 'space/STEP%.4d'%step +".png")
    plt.savefig(fname=fig_folder+'/'+ 'space/STEP%.4d'%step +".pdf")
    
    

    # Show the plot
    plt.show()

#%%
step =47
with open(os.path.join(fig_folder,f'data_STEP%.4d'%step+'.pkl'),'rb') as f:
    data=pkl.load(f, encoding="latin1")
plot_3dprocess(data, step, res=100, cmap=get_cmap('Reds'), use_cbar=True)


#%%
files=os.listdir(fig_folder)
files=[file for file in sorted(files) if file.endswith('pkl')]
space_dir = join(fig_folder, 'space')
if not os.path.isdir(space_dir):
    os.mkdir(space_dir)
os.system('rm ' + space_dir + "/*")
for step, file in enumerate(files):
    
    with open(os.path.join(fig_folder,f'data_STEP%.4d'%step+'.pkl'),'rb') as f:
        data=pkl.load(f, encoding="latin1")
    plot_3dprocess(data, step, res=100, cmap=get_cmap('Reds'), use_cbar=True)
    
with imageio.get_writer(f'{space_dir}/mapping.gif', mode='I') as writer:
    for figurename in sorted(os.listdir(space_dir)):
        if figurename.endswith('png'):
            image = imageio.imread(space_dir + '/' + figurename)
            writer.append_data(image)
print( f'Finish making a gif: {space_dir}/mapping.gif')
#%%