#%%
"""
Created on 2022/07/28
original: gen_openmc_data_discrete_2x2_v1.py, gen_filterlayer_2x2_v1.1.py

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


def gen_materials_geometry_tallies(a_num, panel_density):
    panel = openmc.Material(name='CdZnTe')
    panel.set_density('g/cm3', panel_density)
    panel.add_nuclide('Cd114', 33, percent_type='ao')
    panel.add_nuclide('Zn64', 33, percent_type='ao')
    panel.add_nuclide('Te130', 33, percent_type='ao')

    insulator = openmc.Material(name='Zn')
    insulator.set_density('g/cm3', 1)
    insulator.add_nuclide('Pb208', 11.35)
    outer = openmc.Material(name='Outer_CdZnTe')
    outer.set_density('g/cm3', panel_density)
    outer.add_nuclide('Cd114', 33, percent_type='ao')
    outer.add_nuclide('Zn64', 33, percent_type='ao')
    outer.add_nuclide('Te130', 33, percent_type='ao')

    materials = openmc.Materials(materials=[panel, insulator, outer])
    materials.export_to_xml()

    min_x = openmc.XPlane(x0=-100000, boundary_type='transmission')
    max_x = openmc.XPlane(x0=+100000, boundary_type='transmission')
    min_y = openmc.YPlane(y0=-100000, boundary_type='transmission')
    max_y = openmc.YPlane(y0=+100000, boundary_type='transmission')

    min_xx = openmc.XPlane(x0=-100100, boundary_type='vacuum')
    max_xx = openmc.XPlane(x0=+100100, boundary_type='vacuum')
    min_yy = openmc.YPlane(y0=-100100, boundary_type='vacuum')
    max_yy = openmc.YPlane(y0=+100100, boundary_type='vacuum')    

    #for S1 layer
    min_x1 = openmc.XPlane(x0=-0.4, boundary_type='transmission')   #?
    max_x1 = openmc.XPlane(x0=+0.4, boundary_type='transmission')   #?
    min_y1 = openmc.YPlane(y0=-0.4, boundary_type='transmission')   #?
    max_y1 = openmc.YPlane(y0=+0.4, boundary_type='transmission')   #?

    #for S2 layer
    min_x2 = openmc.XPlane(x0=-0.5, boundary_type='transmission')   #?
    max_x2 = openmc.XPlane(x0=+0.5, boundary_type='transmission')   #?
    min_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')   #?
    max_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')   #?

    #for S3 layer
    min_x3 = openmc.XPlane(x0=-a_num/2, boundary_type='transmission')   #?
    max_x3 = openmc.XPlane(x0=+a_num/2, boundary_type='transmission')   #?
    min_y3 = openmc.YPlane(y0=-a_num/2, boundary_type='transmission')   #?
    max_y3 = openmc.YPlane(y0=+a_num/2, boundary_type='transmission')   #?

    #s1 region
    s1_region = +min_x1 & -max_x1 & +min_y1 & -max_y1   #?
    #s2 region
    s2_region = +min_x2 & -max_x2 & +min_y2 & -max_y2   #?
    #s3 region
    s3_region = +min_x3 & -max_x3 & +min_y3 & -max_y3   #?
    #s4 region
    s4_region = +min_x & -max_x & +min_y & -max_y
    #s5 region
    s5_region = +min_xx & -max_xx & +min_yy & -max_yy
    #define s1 cell
    s1_cell = openmc.Cell(name='s1 cell', fill=panel, region=s1_region)
    #define s2 cell
    s2_cell = openmc.Cell(name='s2 cell', fill=insulator, region= ~s1_region & s2_region)
    # Create a Universe to encapsulate a fuel pin
    cell_universe = openmc.Universe(name='universe', cells=[s1_cell, s2_cell])   #?
    # Create fuel assembly Lattice
    assembly = openmc.RectLattice(name='detector arrays')
    assembly.pitch = (1, 1)
    assembly.lower_left = [-1 * a_num / 2.0] * 2   #?
    assembly.universes = [[cell_universe] * a_num] * a_num   #?
    # Create root Cell
    arrays_cell = openmc.Cell(name='arrays cell', fill=assembly, region = s3_region)   #?
    root_cell = openmc.Cell(name='root cell', fill=None, region = ~s3_region & s4_region)   #?
    outer_cell = openmc.Cell(name='outer cell', fill=None, region = ~s4_region & s5_region)

    root_universe = openmc.Universe(name='root universe')
    root_universe.add_cell(arrays_cell)   #?
    root_universe.add_cell(root_cell)   #?
    root_universe.add_cell(outer_cell)

    root_universe.plot(width=(22, 22), basis='xy')
    plt.show()
    plt.savefig('save_fig/geometry.png')
    plt.savefig('save_fig/geometry.pdf')
    plt.close()

    # Create Geometry and export to "geometry.xml"
    geometry = openmc.Geometry(root_universe)
    geometry.export_to_xml()
    # Instantiate an empty Tallies object
    tallies = openmc.Tallies()

    # Instantiate a tally Mesh
    mesh = openmc.RegularMesh(mesh_id=1)
    mesh.dimension = [a_num, a_num]   #?
    mesh.lower_left = [-a_num/2, -a_num/2]   #?
    mesh.width = [1, 1]
    # Instantiate tally Filter
    mesh_filter = openmc.MeshFilter(mesh)
    # Instantiate the Tally
    tally = openmc.Tally(name='mesh tally')

    #else:
    if True:
        tally.filters = [mesh_filter]

    tally.scores = ["absorption"]

    # Add mesh and Tally to Tallies
    tallies.append(tally)

    # Instantiate tally Filter
    cell_filter = openmc.CellFilter(s1_cell)

    # Instantiate the tally
    tally = openmc.Tally(name='cell tally')
    tally.filters = [cell_filter]
    tally.scores = ['absorption']
    tally.nuclides = ['Cd114', 'Te130', 'Zn64']

    # Instantiate tally Filter
    distribcell_filter = openmc.DistribcellFilter(s2_cell)

    # Instantiate tally Trigger for kicks
    trigger = openmc.Trigger(trigger_type='std_dev', threshold=5e-5)
    trigger.scores = ['absorption']

    # Instantiate the Tally
    tally = openmc.Tally(name='distribcell tally')
    tally.filters = [distribcell_filter]
    tally.scores = ['absorption']
    tally.nuclides = ['Cd114', 'Te130', 'Zn64']
    tally.triggers = [trigger]

    # Export to "tallies.xml"
    tallies.export_to_xml()
    
    # Remove old HDF5 (summary, statepoint) files
    os.system('rm statepoint.*')
    os.system('rm summary.*')

def gen_settings(src_energy, src_strength, en_prob, num_particles, batch_size, sources):
    num_sources = len(sources)
    sources_list = []
    for i in range(num_sources):
        point = openmc.stats.Point((sources[i]['position'][0], sources[i]['position'][1], 0))
        source = openmc.Source(space=point, particle='photon', energy=src_energy, strength=src_strength)  #!20220204    #!20220118
        source.energy = openmc.stats.Discrete(x=(sources[i]['counts']), p=en_prob)
        sources_list.append(source) #, source2, source3]     #!20220118

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.photon_transport = True
    settings.source = sources_list
    settings.batches = batch_size
    settings.inactive = 10
    settings.particles = num_particles

    settings.export_to_xml()

def run_openmc():
    # Run OpenMC!
    openmc.run()


def process_aft_openmc(a_num, folder1, file1, folder2, file2, sources, seg_angles, norm):
    statepoints = glob.glob('statepoint.*.h5')
    sp = openmc.StatePoint(statepoints[-1])
    tally = sp.get_tally(name='mesh tally')
    data = tally.get_values()
    df = tally.get_pandas_dataframe(nuclides=False)
    pd.options.display.float_format = '{:.2e}'.format
    fiss = df[df['score'] == 'absorption']
    mean = fiss['mean'].values.reshape((a_num, a_num))
    mean = np.transpose(mean)
    max = mean.max()
    if norm:
        mean_me = mean.mean()
        mean_st = mean.std()
        mean = (mean-mean_me)/mean_st
    absorb = tally.get_slice(scores=['absorption'])
    stdev = absorb.std_dev.reshape((a_num, a_num))
    stdev_max = stdev.max()

    num_sources = len(sources)
    
    data_json={}
    data_json['source']=sources
    data_json['intensity']=100
    data_json['miu_detector']=0.3
    data_json['miu_medium']=1.2
    data_json['miu_air']=0.00018

    if num_sources==1:
        data_json['output']=get_output(sources, seg_angles).tolist()
    else:
        data_json['output']=get_output_mul(sources, seg_angles).tolist()
    data_json['num_sources']=num_sources
    data_json['seg_angles']=seg_angles
    data_json['miu_de']=0.5
    mean_list=mean.T.reshape((1, a_num**2)).tolist()
    data_json['input']=mean_list[0]
    data_json['bean_num']=0.5
    modelinfo={'det_num_x': 10, 'det_num_y': 10, 'det_y': 0.03, 'det_x': 0.03, 'size_x': 0.5, 'size_x': 0.5, 'size_y': 0.5, 'med_margin': 0.0015}    #!20220119 constant!
    data_json['model_info']=modelinfo
    with open(folder1+file1,"w") as f:
        json.dump(data_json, f)

    plt.imshow(mean, interpolation='nearest', cmap='gist_gray')#"plasma")

    ds_ag_list = file2[:-5].split('_')[1:]  #!20220517
    ds_ag_title = ''
    for i in range(num_sources):
        ds, ag = ds_ag_list[2*i], ds_ag_list[2*i+1]
        ds_ag_line = f'dist{i}: {ds},  angle{i}: {ag}'
        if i != num_sources-1:
            ds_ag_line += '\n'
        ds_ag_title += ds_ag_line
    plt.title(ds_ag_title)
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.savefig(folder2 + file2)
    plt.savefig(folder2 + file2[:-3] + 'pdf')
    plt.close()
    print('json dir')
    print(folder1+file1)
    print('fig dir')
    print(folder2+file2)
    return mean

def get_output(source, num):
    sec_center=np.linspace(-np.pi,np.pi,num+1)
    output=np.zeros(num)#(40)
    sec_dis=2*np.pi/num #40.
    angle=np.arctan2(source[0]["position"][1],source[0]["position"][0])
    before_indx=int((angle+np.pi)/sec_dis)
    if before_indx>=num:
        before_indx-=num
    after_indx=before_indx+1
    if after_indx>=num:
        after_indx-=num
    w1=abs(angle-sec_center[before_indx])
    w2=abs(angle-sec_center[after_indx])
    if w2>sec_dis:
        w2=abs(angle-(sec_center[after_indx]+2*np.pi))
    output[before_indx]+=w2/(w1+w2)
    output[after_indx]+=w1/(w1+w2)
    return output

def get_output_mul(sources, num):
    sec_center=np.linspace(-np.pi,np.pi,num+1)
    output=np.zeros(num)
    sec_dis=2*np.pi/num
    ws=np.array([np.mean(source["counts"]) for source in sources])
    ws=ws/ws.sum()
    for i in range(len(sources)):
        source =sources[i]
        angle=np.arctan2(source["position"][1],source["position"][0])
        before_indx=int((angle+np.pi)/sec_dis)
        after_indx=before_indx+1
        if after_indx>=num:
            after_indx-=num
        w1=abs(angle-sec_center[before_indx])
        w2=abs(angle-sec_center[after_indx])
        if w2>sec_dis:
            w2=abs(angle-(sec_center[after_indx]+2*np.pi))
        output[before_indx]+=w2/(w1+w2)*ws[i]
        output[after_indx]+=w1/(w1+w2)*ws[i]
    return output

def before_openmc(a_num, sources_d_th, num_particles, seg_angles):
    batches = 100
    panel_density = 5.76
    src_E = None
    src_Str = 10
    num_sources = len(sources_d_th)
    energy_prob = (1)
    start = timeit.timeit()
    start_time = datetime.now()
    gen_materials_geometry_tallies(a_num, panel_density)
    j=batches
    sources = []
    for i in range(num_sources):
        theta=sources_d_th[i][1]*np.pi/180
        dist = sources_d_th[i][0]
        source = {}
        src_xy = [float(dist*np.cos(theta)), float(dist*np.sin(theta))]
        source['position']=src_xy
        source['counts']=sources_d_th[i][2]
        sources.append(source)
    if num_sources==1:
        get_output(sources, seg_angles) #!20220803
    else:
        get_output_mul(sources, seg_angles) #!20220803
    gen_settings(src_energy=src_E, src_strength=src_Str, en_prob=energy_prob, num_particles=num_particles, batch_size=j, sources=sources) 

def after_openmc(a_num, sources_d_th, folder1, folder2, seg_angles, header):
    num_sources = len(sources_d_th)
    d_a_seq = ""
    for i in range(num_sources):
        d_a_seq += '_' + str(round(sources_d_th[i][0], 5)) + '_' + str(round(sources_d_th[i][1], 5))
    file1=header + d_a_seq + '.json'    #!20220803
    file2=header + d_a_seq + '.png'

    isExist1 = os.path.exists(folder1)
    if not isExist1:
        os.makedirs(folder1)
        print("The new directory "+ folder1 +" is created!")

    isExist2 = os.path.exists(folder2)
    if not isExist2:
        os.makedirs(folder2)
        print("The new directory "+ folder2 +" is created!")

    sources = []
    for i in range(num_sources):
        theta=sources_d_th[i][1]*np.pi/180
        dist = sources_d_th[i][0]
        source = {}
        src_xy = [float(dist*np.cos(theta)), float(dist*np.sin(theta))]
        source['position']=src_xy
        source['counts']=sources_d_th[i][2]
        sources.append(source)
    mm = process_aft_openmc(a_num, folder1, file1, folder2, file2, sources, seg_angles, norm=True)
    return mm


#%%

if __name__ == '__main__':
    num_sources = 2
    a_num = 2
    num_data = 5000
    seg_angles = 64
    dist_min = 50
    dist_max = 500
    source_energies = [0.5e6, 0.5e6]
    num_particles = 20000
    header = 'data'

    folder1=f'openmc/data_{a_num}x{a_num}_{num_sources}src_{seg_angles}_data_20220812_v1.1/'
    folder2=f'openmc/data_{a_num}x{a_num}_{num_sources}src_{seg_angles}_fig_20220812_v1.1/'

    for i in range(num_data):
        sources_d_th = [[np.random.randint(dist_min, dist_max), float(np.random.randint(0, 360) + np.random.random(1)), source_energies[i]] for i in range(num_sources)]
        for i in range(num_sources):
            print(f"dist {i}: " + str(sources_d_th[i][0]))
            print(f"angle {i}: " + str(sources_d_th[i][1]))
            print(f"energy {i}: " + str(sources_d_th[i][2]))
        before_openmc(a_num, sources_d_th, num_particles, seg_angles)
        run_openmc()
        mm = after_openmc(a_num, sources_d_th, folder1, folder2, seg_angles, header)

# %%
