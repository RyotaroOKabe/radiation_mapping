#%%

"""

ROkabe
last update: 2022/07/05
original: gen_openmc_data_discrete_a3_v4.py

Major update: theta digit: 180 deg = 18*10 deg >>> 20*9 deg

"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json #!20220119
#import torch   #!20220125 20220430 remove for its incompatibility
import timeit   #!20220224
from datetime import datetime   #!20220224
import openmc

GPU_INDEX = 0
USE_CPU = False


#%%
def gen_materials_geometry_tallies(a_num, panel_density, e_filter, *energy):
    panel = openmc.Material(name='CdZnTe')  #!20220302
    panel.set_density('g/cm3', panel_density)#5.8)
    panel.add_nuclide('Cd114', 33, percent_type='ao')
    panel.add_nuclide('Zn64', 33, percent_type='ao')
    panel.add_nuclide('Te130', 33, percent_type='ao')

    insulator = openmc.Material(name='Zn')  #!20220302
    insulator.set_density('g/cm3', 1)
    insulator.add_nuclide('Pb208', 11.35)
    
    outer = openmc.Material(name='Outer_CdZnTe')
    outer.set_density('g/cm3', panel_density)#5.8)
    outer.add_nuclide('Cd114', 33, percent_type='ao')
    outer.add_nuclide('Zn64', 33, percent_type='ao')
    outer.add_nuclide('Te130', 33, percent_type='ao')

    materials = openmc.Materials(materials=[panel, insulator, outer])
    materials.cross_sections = '/home/rokabe/data1/openmc/endfb71_hdf5/cross_sections.xml'
    materials.export_to_xml()

    min_x = openmc.XPlane(x0=-100000, boundary_type='transmission')
    max_x = openmc.XPlane(x0=+100000, boundary_type='transmission')
    min_y = openmc.YPlane(y0=-100000, boundary_type='transmission')
    max_y = openmc.YPlane(y0=+100000, boundary_type='transmission')
    min_z = openmc.ZPlane(z0=-100000, boundary_type='transmission')   #!20220628
    max_z = openmc.ZPlane(z0=+100000, boundary_type='transmission')

    #for S1 layer
    min_x1 = openmc.XPlane(x0=-0.4, boundary_type='transmission')
    max_x1 = openmc.XPlane(x0=+0.4, boundary_type='transmission')
    min_y1 = openmc.YPlane(y0=-0.4, boundary_type='transmission')
    max_y1 = openmc.YPlane(y0=+0.4, boundary_type='transmission')
    min_z1 = openmc.ZPlane(z0=-0.4, boundary_type='transmission')   #!20220628
    max_z1 = openmc.ZPlane(z0=+0.4, boundary_type='transmission')

    #for S2 layer
    min_x2 = openmc.XPlane(x0=-0.5, boundary_type='transmission')
    max_x2 = openmc.XPlane(x0=+0.5, boundary_type='transmission')
    min_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')
    max_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')
    min_z2 = openmc.ZPlane(z0=-0.5, boundary_type='transmission')   #!20220628
    max_z2 = openmc.ZPlane(z0=+0.5, boundary_type='transmission')

    #for S3 layer
    min_x3 = openmc.XPlane(x0=-a_num/2, boundary_type='transmission')   #!20220629 #!a_num
    max_x3 = openmc.XPlane(x0=+a_num/2, boundary_type='transmission')
    min_y3 = openmc.YPlane(y0=-a_num/2, boundary_type='transmission')
    max_y3 = openmc.YPlane(y0=+a_num/2, boundary_type='transmission')
    min_z3 = openmc.ZPlane(z0=-a_num/2, boundary_type='transmission')
    max_z3 = openmc.ZPlane(z0=+a_num/2, boundary_type='transmission')
    
    min_xx = openmc.XPlane(x0=-100100, boundary_type='vacuum')
    max_xx = openmc.XPlane(x0=+100100, boundary_type='vacuum')
    min_yy = openmc.YPlane(y0=-100100, boundary_type='vacuum')
    max_yy = openmc.YPlane(y0=+100100, boundary_type='vacuum')
    min_zz = openmc.ZPlane(z0=-100100, boundary_type='vacuum')   #!20220628
    max_zz = openmc.ZPlane(z0=+100100, boundary_type='vacuum')

    #s1 region
    s1_region = +min_x1 & -max_x1 & +min_y1 & -max_y1 & +min_z1 & -max_z1  #!20220628

    #s2 region
    s2_region = +min_x2 & -max_x2 & +min_y2 & -max_y2 & +min_z2 & -max_z2  #!20220628

    #s3 region
    s3_region = +min_x3 & -max_x3 & +min_y3 & -max_y3 & +min_z3 & -max_z3  #!20220628

    #s4 region
    s4_region = +min_x & -max_x & +min_y & -max_y & +min_z & -max_z   #!20220628
    
    #s5 region
    s5_region = +min_xx & -max_xx & +min_yy & -max_yy & +min_zz & -max_zz   #!20220628

    #define s1 cell
    s1_cell = openmc.Cell(name='s1 cell', fill=panel, region=s1_region)

    #define s2 cell
    s2_cell = openmc.Cell(name='s2 cell', fill=insulator, region= ~s1_region & s2_region)

    # Create a Universe to encapsulate a fuel pin
    cell_universe = openmc.Universe(name='universe', cells=[s1_cell, s2_cell])   #!20220117

    # Create fuel assembly Lattice
    assembly = openmc.RectLattice(name='detector arrays')
    assembly.pitch = (1, 1, 1) #(1, 1)   #!20220628
    assembly.lower_left = [-a_num/2, -a_num/2, -a_num/2] #[[-1 * 5 / 2.0] * 2 ] * 2   #!20220629 #! a_num
    assembly.universes = [[[cell_universe] * a_num] * a_num] * a_num

    # Create root Cell
    arrays_cell = openmc.Cell(name='arrays cell', fill=assembly, region = s3_region)
    root_cell = openmc.Cell(name='root cell', fill=None, region = ~s3_region & s4_region)   #!20220117
    outer_cell = openmc.Cell(name='outer cell', fill=None, region = ~s4_region & s5_region)   #!20220124

    root_universe = openmc.Universe(name='root universe')
    root_universe.add_cell(arrays_cell)
    root_universe.add_cell(root_cell)
    root_universe.add_cell(outer_cell)

    root_universe.plot(width=(22, 22), basis='xy')     #!20220124
    plt.show()
    plt.savefig('save_fig/geometry_xy_20220628.png')   #!20220628
    plt.close()

    root_universe.plot(width=(22, 22), basis='yz')     #!20220124
    plt.show()
    plt.savefig('save_fig/geometry_yz_20220628.png')   #!20220628
    plt.close()

    root_universe.plot(width=(22, 22), basis='xz')     #!20220124
    plt.show()
    plt.savefig('save_fig/geometry_zx_20220628.png')   #!20220628
    plt.close()

    # Create Geometry and export to "geometry.xml"
    geometry = openmc.Geometry(root_universe)
    geometry.export_to_xml()

    # Instantiate an empty Tallies object
    tallies = openmc.Tallies()

    # Instantiate a tally Mesh
    mesh = openmc.RegularMesh(mesh_id=1)    #!20220628
    mesh.dimension = [a_num, a_num, a_num] #[10, 10]
    mesh.lower_left = [-a_num/2, -a_num/2, -a_num/2] #[-5, -5]  #[-10, -10] #!20220629 #!a_num
    mesh.width = [1, 1, 1] #[2, 2]

    mesh_filter = openmc.MeshFilter(mesh)

    tally = openmc.Tally(name='mesh tally')
    
    if e_filter:
        energy_filter = openmc.EnergyFilter(*energy)    #!20220204
        tally.filters = [mesh_filter, energy_filter]    #!20220204
    
    else:
        tally.filters = [mesh_filter]    #!20220204

    tally.scores = ["absorption"]   #, 'fission', 'nu-fission'] #!20220117

    # Add mesh and Tally to Tallies
    tallies.append(tally)

    # Instantiate tally Filter
    cell_filter = openmc.CellFilter(s1_cell)

    # Instantiate the tally
    tally = openmc.Tally(name='cell tally')
    tally.filters = [cell_filter]
    tally.scores = ['absorption']#['scatter']
    tally.nuclides = ['Cd114', 'Te130', 'Zn64']

    # Instantiate tally Filter
    distribcell_filter = openmc.DistribcellFilter(s2_cell)

    # Instantiate tally Trigger for kicks
    trigger = openmc.Trigger(trigger_type='std_dev', threshold=5e-5)
    trigger.scores = ['absorption']

    # Instantiate the Tally
    tally = openmc.Tally(name='distribcell tally')
    tally.filters = [distribcell_filter]
    tally.scores = ['absorption'] #['absorption', 'scatter']
    tally.nuclides = ['Cd114', 'Te130', 'Zn64']  #!20220117
    tally.triggers = [trigger]

    # Export to "tallies.xml"
    tallies.export_to_xml()

    os.system('rm statepoint.*')
    os.system('rm summary.*')


#def gen_settings(rad_source1=rad_source_x):
def gen_settings(src_energy=None, src_strength=1, en_source=1e6, en_prob=1, num_particles=10000, batch_size=100, source_x=100, source_y=100, source_z=100): #!20220628
    point1 = openmc.stats.Point((source_x, source_y, source_z)) #!20220628
    source1 = openmc.Source(space=point1, particle='photon', energy=src_energy, strength=src_strength)  #!20220204    #!20220118

    source1.energy = openmc.stats.Discrete(x=en_source, p=en_prob)

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.photon_transport = True
    settings.source = [source1]
    settings.batches = batch_size
    settings.inactive = 10
    settings.particles = num_particles
    settings.export_to_xml()

def run_openmc():
    openmc.run()

def process_aft_openmc_v1(a_num, folder1='random_savearray/', file1='detector_1source_20220118.txt', \
                        folder2='random_savefig/', file2='detector_1source_20220118.png',\
                            source_x=100, source_y=100, source_z=100, ph_num=40, th_num=18, norm=True):
    statepoints = glob.glob('statepoint.*.h5')
    sp = openmc.StatePoint(statepoints[-1])
    tally = sp.get_tally(name='mesh tally')
    df = tally.get_pandas_dataframe(nuclides=False)
    pd.options.display.float_format = '{:.2e}'.format
    fiss = df[df['score'] == 'absorption']
    mean = fiss['mean'].values.reshape((a_num,a_num,a_num))
    mean = np.transpose(mean)
    max = mean.max()
    min = mean.min()
    if norm:
        mean_me = mean.mean()
        mean_st = mean.std()
        mean = (mean-mean_me)/mean_st

    absorb = tally.get_slice(scores=['absorption'])
    stdev = absorb.std_dev.reshape((a_num,a_num,a_num))
    stdev_max = stdev.max()
    output_ph, output_th = get_output([source_x, source_y, source_z], ph_num, th_num)
    output_mtrx = np.einsum('ij,jk->ik', output_th.reshape(-1, 1), output_ph.reshape(1, -1))

    ds, ph, th = file2[:-5].split('_')[1:]
    #==================================
    
    data_json={}
    data_json['source']=[source_x, source_y, source_z]
    data_json['a_num'] = a_num
    data_json['r'] = ds
    data_json['phi'] = ph
    data_json['theta'] = th
    data_json['intensity']=100
    data_json['miu_detector']=0.3
    data_json['miu_medium']=1.2
    data_json['miu_air']=0.00018
    data_json['output_ph']=output_ph.tolist()
    data_json['output_th']=output_th.tolist()
    data_json['output_mtrx']=output_mtrx.tolist()
    data_json['ph_num']=ph_num
    data_json['th_num']=th_num
    data_json['miu_de']=0.5
    mean_list=mean.T.reshape((1, a_num**3)).tolist()
    data_json['input']=mean_list[0]
    data_json['bean_num']=0.5
    modelinfo={'det_num_x': 10, 'det_num_y': 10, 'det_y': 0.03, 'det_x': 0.03, 'size_x': 0.5, 'size_x': 0.5, 'size_y': 0.5, 'med_margin': 0.0015}    #!20220119 constant!
    data_json['model_info']=modelinfo
    with open(folder1+file1,"w") as f:
        json.dump(data_json, f)
    maxmax = mean.max()
    minmin = mean.min()

    axes=[]
    fig, axs = plt.subplots(3, a_num, figsize=(a_num*5,15), constrained_layout=True)  #!20220629 #!a_num
    fs_label = 20
    fs_title = 22
    fs_tick = 18
    for xa in range(a_num):
        ax = axs[0, xa]
        xslice = ax.imshow(mean[xa, :, :], vmin=minmin, vmax=maxmax, interpolation='nearest', cmap="plasma")       #!20220118
        ax.set_xlabel('z', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
        ax.set_ylabel('y', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
        ax.tick_params(axis='x', labelsize=fs_tick)
        ax.tick_params(axis='y', labelsize=fs_tick)
        ax.set_title("X=" + str(xa), fontsize = fs_title)
    for ya in range(a_num):
        ax = axs[1, ya]
        yslice = ax.imshow(mean[:, ya, :], vmin=minmin, vmax=maxmax, interpolation='nearest', cmap="plasma")       #!20220118
        ax.set_xlabel('z', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
        ax.set_ylabel('x', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
        ax.tick_params(axis='x', labelsize=fs_tick)
        ax.tick_params(axis='y', labelsize=fs_tick)
        ax.set_title("Y=" + str(ya), fontsize = fs_title)
    for za in range(a_num):
        ax = axs[2, za]
        zslice = ax.imshow(mean[:, :, za], vmin=minmin, vmax=maxmax, interpolation='nearest', cmap="plasma")       #!20220118
        ax.set_xlabel('y', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
        ax.set_ylabel('x', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
        ax.tick_params(axis='x', labelsize=fs_tick)
        ax.tick_params(axis='y', labelsize=fs_tick)
        ax.set_title("Z=" + str(za), fontsize = fs_title)

    fig.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.8, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.3)
    
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(zslice, ax=axs[:, -1], cax=cbar_ax)
    #?fig.suptitle('dist: ' + ds + ',  \u03C6: ' + ph + ',  \u03B8: ' + th + '\nMean_max: ' + str(max) + '\nStdev_max: ' + str(stdev_max), fontsize=25)
    fig.suptitle('r: ' + ds + ' /  \u03C6: ' + ph + ' /  \u03B8: ' + th, fontsize=30)
    fig.savefig(folder2 + file2)
    plt.close()

    print('json dir')
    print(folder1+file1)

    print('fig dir')
    print(folder2+file2)
    return mean #!20220331

def get_output(source, ph_num, th_num): #!20220706
    sec_center=np.linspace(-np.pi,np.pi,ph_num+1)
    sec_th=np.linspace(np.pi/(2*th_num),np.pi*(2*th_num-1)/(2*th_num),th_num) # sec_th=np.linspace(np.pi/36,np.pi*35/36,18)
    output_ph=np.zeros(ph_num)
    output_th=np.zeros(th_num)  #output_th=np.zeros(18)
    sec_dis_ph=2*np.pi/ph_num
    sec_dis_th=np.pi/th_num    #sec_dis_th=np.pi/18.
    angle_ph=np.arctan2(source[1],source[0])
    angle_th=np.arctan2(np.sqrt(source[0]**2+source[1]**2), source[2])
    before_indx=int((angle_ph+np.pi)/sec_dis_ph)
    if before_indx>=ph_num:
        before_indx-=ph_num
    after_indx=before_indx+1
    if after_indx>=ph_num:
        after_indx-=ph_num
    w1=abs(angle_ph-sec_center[before_indx])
    w2=abs(angle_ph-sec_center[after_indx])
    if w2>sec_dis_ph:
        w2=abs(angle_ph-(sec_center[after_indx]+2*np.pi))
    output_ph[before_indx]+=w2/(w1+w2)
    output_ph[after_indx]+=w1/(w1+w2)
    
    before_indx_th=int(angle_th/sec_dis_th)
    if before_indx_th>=th_num:
        before_indx_th=2*th_num-1-before_indx_th
    after_indx_th=before_indx_th+1
    if after_indx_th>=th_num:
        after_indx_th=2*th_num-2-after_indx_th
    w1_th=abs(angle_th-sec_th[before_indx_th])
    w2_th=abs(angle_th-sec_th[after_indx_th])
    if w2_th>sec_dis_th:
        w2_th=abs(angle_th-(sec_th[after_indx_th]+2*np.pi))
    output_th[before_indx_th]+=w2_th/(w1_th+w2_th)
    output_th[after_indx_th]+=w1_th/(w1_th+w2_th)
    return output_ph, output_th

def before_openmc(a_num, rad_dist, rad_phi, rad_th, ph_num, th_num, num_particles):
    batches = 100
    panel_density = 5.76 #g/cm3
    src_E = None    #[1,3]
    src_Str = 10
    energy_filter_range = [0.1e6, 2e6]     #!20220223
    e_filter_tf=False
    source_energy = (0.5e6)
    energy_prob = (1)
    start = timeit.timeit()
    start_time = datetime.now()
    
    gen_materials_geometry_tallies(a_num, panel_density, e_filter_tf, energy_filter_range)     #!20220205
    j=batches
    phi=rad_phi*np.pi/180   #!20220628
    theta = rad_th*np.pi/180
    rad_x, rad_y, rad_z=[float(rad_dist*np.cos(phi)*np.sin(theta)), float(rad_dist*np.sin(phi)*np.sin(theta)), float(rad_dist*np.cos(theta))]   #!20220628
    print([rad_x, rad_y, rad_z])
    #?get_output([rad_x, rad_y, rad_z])   #!20220628
    # output_ph, output_th = get_output([rad_x, rad_y, rad_z], ph_num, th_num)   #!20220706
    gen_settings(src_energy=src_E, src_strength=src_Str,  en_source=source_energy, en_prob=energy_prob, num_particles=num_particles, batch_size=j, source_x=rad_x, source_y=rad_y, source_z=rad_z)    #!20220224

def after_openmc(a_num, rad_dist, rad_phi, rad_th, ph_num, th_num, folder1, folder2, header):      
    file1=header + "_" + str(round(rad_dist, 5)) + '_' + str(round(rad_phi, 5)) + '_' + str(round(rad_th, 5)) +  '.json'
    file2=header + "_" + str(round(rad_dist, 5)) + '_' + str(round(rad_phi, 5)) + '_' + str(round(rad_th, 5)) + '.png'
    isExist1 = os.path.exists(folder1)
    if not isExist1:
        os.makedirs(folder1)
        print("The new directory "+ folder1 +" is created!")
    isExist2 = os.path.exists(folder2)
    if not isExist2:
        os.makedirs(folder2)
        print("The new directory "+ folder2 +" is created!")
    
    phi=rad_phi*np.pi/180
    theta=rad_th*np.pi/180
    rad_x, rad_y, rad_z=[float(rad_dist*np.cos(phi)*np.sin(theta)), float(rad_dist*np.sin(phi)*np.sin(theta)), float(rad_dist*np.cos(theta))]   #!20220119
            
    mm = process_aft_openmc_v1(a_num, folder1, file1, folder2, file2, rad_x, rad_y, rad_z, ph_num, th_num, norm=True)  #!20220201 #!20220119

    return mm #!20220508

#%%
if __name__ == '__main__':
    a_num = 4
    num_data = 3500
    dist = 50
    num_particles =100000
    dist_min = 100
    dist_max = 100
    ph_num = 32
    th_num = 16
    folder1=f'openmc/discrete_data_20220707_{a_num}^3_v1/'
    folder2=f'openmc/discrete_fig_20220707_{a_num}^3_v1/'
    header = "data"

    for i in range(num_data):
        rad_phi=float(np.random.randint(0, 360) + np.random.random(1))
        rad_th=float(np.random.randint(0, 180) + np.random.random(1))
        print("dist: " + str(dist))
        print("angle: " + str(rad_phi))
        before_openmc(a_num, dist, rad_phi, rad_th, ph_num, th_num, num_particles)  #!20220628 #!a_num
        openmc.run()
        mm = after_openmc(a_num, dist, rad_phi, rad_th, ph_num, th_num, folder1, folder2, header) #!20220706

# %%
