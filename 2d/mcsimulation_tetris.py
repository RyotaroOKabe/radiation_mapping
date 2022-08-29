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


def gen_materials_geometry_tallies(use_panels, panel_density):
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

    # for A inner layer
    Amin_x1 = openmc.XPlane(x0=+0.1, boundary_type='transmission')   #!20220720
    Amax_x1 = openmc.XPlane(x0=+0.9, boundary_type='transmission')
    Amin_y1 = openmc.YPlane(y0=+0.6, boundary_type='transmission')
    Amax_y1 = openmc.YPlane(y0=+1.4, boundary_type='transmission')

    #for A outer layer
    Amin_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')   #!20220720
    Amax_x2 = openmc.XPlane(x0=+1.0, boundary_type='transmission')
    Amin_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')
    Amax_y2 = openmc.YPlane(y0=+1.5, boundary_type='transmission')

    # for B inner layer (y-1 from A)
    Bmin_x1 = openmc.XPlane(x0=+0.1, boundary_type='transmission')   #!20220720
    Bmax_x1 = openmc.XPlane(x0=+0.9, boundary_type='transmission')
    Bmin_y1 = openmc.YPlane(y0=-0.4, boundary_type='transmission')
    Bmax_y1 = openmc.YPlane(y0=+0.4, boundary_type='transmission')

    #for B outer layer (y-1 from A)
    Bmin_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')   #!20220720
    Bmax_x2 = openmc.XPlane(x0=+1.0, boundary_type='transmission')
    Bmin_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')
    Bmax_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')

    # for C inner layer (y-2 from A)
    Cmin_x1 = openmc.XPlane(x0=+0.1, boundary_type='transmission')   #!2022020
    Cmax_x1 = openmc.XPlane(x0=+0.9, boundary_type='transmission')
    Cmin_y1 = openmc.YPlane(y0=-1.4, boundary_type='transmission')
    Cmax_y1 = openmc.YPlane(y0=-0.6, boundary_type='transmission')

    #for C outer layer (y-2 from A)
    Cmin_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')   #!20220720
    Cmax_x2 = openmc.XPlane(x0=+1.0, boundary_type='transmission')
    Cmin_y2 = openmc.YPlane(y0=-1.5, boundary_type='transmission')
    Cmax_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')

    # for D inner layer (x-1 from A)
    Dmin_x1 = openmc.XPlane(x0=-0.9, boundary_type='transmission')   #!20220720
    Dmax_x1 = openmc.XPlane(x0=-0.1, boundary_type='transmission')
    Dmin_y1 = openmc.YPlane(y0=+0.6, boundary_type='transmission')
    Dmax_y1 = openmc.YPlane(y0=+1.4, boundary_type='transmission')

    #for D outer layer (x-1 from A)
    Dmin_x2 = openmc.XPlane(x0=-1.0, boundary_type='transmission')   #!20220720
    Dmax_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')
    Dmin_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')
    Dmax_y2 = openmc.YPlane(y0=+1.5, boundary_type='transmission')

    # for E inner layer (x-1, y-1 from A)
    Emin_x1 = openmc.XPlane(x0=-0.9, boundary_type='transmission')   #!20220720
    Emax_x1 = openmc.XPlane(x0=+0.9, boundary_type='transmission')
    Emin_y1 = openmc.YPlane(y0=-0.4, boundary_type='transmission')
    Emax_y1 = openmc.YPlane(y0=+0.4, boundary_type='transmission')

    #for E outer layer (x-1, y-1 from A)
    Emin_x2 = openmc.XPlane(x0=-1.0, boundary_type='transmission')   #!20220720
    Emax_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')
    Emin_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')
    Emax_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')

    # for E inner layer (x-1, y-2 from A)
    Fmin_x1 = openmc.XPlane(x0=-0.9, boundary_type='transmission')   #!2022020
    Fmax_x1 = openmc.XPlane(x0=+0.9, boundary_type='transmission')
    Fmin_y1 = openmc.YPlane(y0=-1.4, boundary_type='transmission')
    Fmax_y1 = openmc.YPlane(y0=-0.6, boundary_type='transmission')

    #for F outer layer (x-1, y-2 from A)
    Fmin_x2 = openmc.XPlane(x0=-1.0, boundary_type='transmission')   #!20220720
    Fmax_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')
    Fmin_y2 = openmc.YPlane(y0=-1.5, boundary_type='transmission')
    Fmax_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')

    #for S3 layer
    min_x3 = openmc.XPlane(x0=-1.0, boundary_type='transmission')     #!20220715 size-change!
    max_x3 = openmc.XPlane(x0=+1.0, boundary_type='transmission')
    min_y3 = openmc.YPlane(y0=-1.5, boundary_type='transmission')
    max_y3 = openmc.YPlane(y0=+1.5, boundary_type='transmission')

    # A1 region
    A1_region = +Amin_x1 & -Amax_x1 & +Amin_y1 & -Amax_y1
    # A2 region
    A2_region = +Amin_x2 & -Amax_x2 & +Amin_y2 & -Amax_y2
    # A1 region
    B1_region = +Bmin_x1 & -Bmax_x1 & +Bmin_y1 & -Bmax_y1
    # A2 region
    B2_region = +Bmin_x2 & -Bmax_x2 & +Bmin_y2 & -Bmax_y2
    # A1 region
    C1_region = +Cmin_x1 & -Cmax_x1 & +Cmin_y1 & -Cmax_y1
    # A2 region
    C2_region = +Cmin_x2 & -Cmax_x2 & +Cmin_y2 & -Cmax_y2
    # D1 region
    D1_region = +Dmin_x1 & -Dmax_x1 & +Dmin_y1 & -Dmax_y1
    # D2 region
    D2_region = +Dmin_x2 & -Dmax_x2 & +Dmin_y2 & -Dmax_y2
    # E1 region
    E1_region = +Emin_x1 & -Emax_x1 & +Emin_y1 & -Emax_y1
    # E2 region
    E2_region = +Emin_x2 & -Emax_x2 & +Emin_y2 & -Emax_y2
    # F1 region
    F1_region = +Fmin_x1 & -Fmax_x1 & +Fmin_y1 & -Fmax_y1
    # F2 region
    F2_region = +Fmin_x2 & -Fmax_x2 & +Fmin_y2 & -Fmax_y2
    #s3 region
    s3_region = +min_x3 & -max_x3 & +min_y3 & -max_y3   #?
    #s4 region
    s4_region = +min_x & -max_x & +min_y & -max_y
    #s5 region
    s5_region = +min_xx & -max_xx & +min_yy & -max_yy

    #define A1, A2 cell
    A1_cell = openmc.Cell(name='A1 cell', fill=panel, region=A1_region)
    A2_cell = openmc.Cell(name='A2 cell', fill=insulator, region= ~A1_region & A2_region)
    A2_cell_vac = openmc.Cell(name='A2_vacant cell', fill=None, region=A2_region)
    #define B1, B2 cell
    B1_cell = openmc.Cell(name='B1 cell', fill=panel, region=B1_region)
    B2_cell = openmc.Cell(name='B2 cell', fill=insulator, region= ~B1_region & B2_region)
    B2_cell_vac = openmc.Cell(name='B2_vacant cell', fill=None, region=B2_region)
    #define C1, C2 cell
    C1_cell = openmc.Cell(name='C1 cell', fill=panel, region=C1_region)
    C2_cell = openmc.Cell(name='C2 cell', fill=insulator, region= ~C1_region & C2_region)
    C2_cell_vac = openmc.Cell(name='C2_vacant cell', fill=None, region=C2_region)
    #define D1, D2 cell
    D1_cell = openmc.Cell(name='D1 cell', fill=panel, region=D1_region)
    D2_cell = openmc.Cell(name='D2 cell', fill=insulator, region= ~D1_region & D2_region)
    D2_cell_vac = openmc.Cell(name='D2_vacant cell', fill=None, region=D2_region)
    #define E1, E2 cell
    E1_cell = openmc.Cell(name='E1 cell', fill=panel, region=E1_region)
    E2_cell = openmc.Cell(name='E2 cell', fill=insulator, region= ~E1_region & E2_region)
    E2_cell_vac = openmc.Cell(name='E2_vacant cell', fill=None, region=E2_region)
    #define F1, F2 cell
    F1_cell = openmc.Cell(name='F1 cell', fill=panel, region=F1_region)
    F2_cell = openmc.Cell(name='F2 cell', fill=insulator, region= ~F1_region & F2_region)
    F2_cell_vac = openmc.Cell(name='F2_vacant cell', fill=None, region=F2_region)

    root_cell = openmc.Cell(name='root cell', fill=None, region = ~s3_region & s4_region)
    outer_cell = openmc.Cell(name='outer cell', fill=None, region = ~s4_region & s5_region)
    root_universe = openmc.Universe(name='root universe')
    root_universe.add_cell(root_cell)
    root_universe.add_cell(outer_cell)
    if 'a' in use_panels:
        root_universe.add_cell(A1_cell)
        root_universe.add_cell(A2_cell)
    else:
        root_universe.add_cell(A2_cell_vac)
    if 'b' in use_panels:
        root_universe.add_cell(B1_cell)
        root_universe.add_cell(B2_cell)
    else:
        root_universe.add_cell(B2_cell_vac)
    if 'c' in use_panels:
        root_universe.add_cell(C1_cell)
        root_universe.add_cell(C2_cell)
    else:
        root_universe.add_cell(C2_cell_vac)
    if 'd' in use_panels:
        root_universe.add_cell(D1_cell)
        root_universe.add_cell(D2_cell)
    else:
        root_universe.add_cell(D2_cell_vac)
    if 'e' in use_panels:
        root_universe.add_cell(E1_cell)
        root_universe.add_cell(E2_cell)
    else:
        root_universe.add_cell(E2_cell_vac)
    if 'f' in use_panels:
        root_universe.add_cell(F1_cell)
        root_universe.add_cell(F2_cell)
    else:
        root_universe.add_cell(F2_cell_vac)

    s = ''.join(use_panels) 
    root_universe.plot(width=(20, 20), basis='xy')
    plt.show()
    plt.savefig(f'save_fig/geometry_tetris_{s}_v1.png')
    plt.savefig(f'save_fig/geometry_tetris_{s}_v1.png')
    plt.close()

    # Create Geometry and export to "geometry.xml"
    geometry = openmc.Geometry(root_universe)
    geometry.export_to_xml()
    # Instantiate an empty Tallies object
    tallies = openmc.Tallies()

    # Instantiate a tally Mesh
    mesh = openmc.RegularMesh(mesh_id=1)
    mesh.dimension = [2, 3]
    mesh.lower_left = [-1.0, -1.5]
    mesh.width = [1, 1]
    mesh_filter = openmc.MeshFilter(mesh)
    # Instantiate the Tally
    tally = openmc.Tally(name='mesh tally')
    tally.filters = [mesh_filter]
    tally.scores = ["absorption"]
    # Add mesh and Tally to Tallies
    tallies.append(tally)

    # Export to "tallies.xml"
    tallies.export_to_xml()
    
    # Remove old HDF5 (summary, statepoint) files
    os.system('rm statepoint.*')
    os.system('rm summary.*')

#def gen_settings(src_energy, src_strength, en_prob, num_particles, batch_size, sources):
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


def process_aft_openmc(use_panels, folder1, file1, folder2, file2, sources, seg_angles, norm):
    statepoints = glob.glob('statepoint.*.h5')
    sp = openmc.StatePoint(statepoints[-1])
    tally = sp.get_tally(name='mesh tally')
    data = tally.get_values()
    df = tally.get_pandas_dataframe(nuclides=False)
    pd.options.display.float_format = '{:.2e}'.format
    fiss = df[df['score'] == 'absorption']
    mean = fiss['mean'].values.reshape((3, 2))
    remove_panels = ['a', 'b', 'c', 'd', 'e', 'f']
    for mark in use_panels: #!20220716
        remove_panels.remove(mark)
    print(remove_panels)

    for mark in remove_panels: #!20220716
        id0, id1 = get_position_from_panelID(mark)
        mean[id0, id1] = 0
    mean = np.transpose(mean)
    max = mean.max()
    if norm:
        mean_me = mean.mean()
        mean_st = mean.std()
        mean = (mean-mean_me)/mean_st
    absorb = tally.get_slice(scores=['absorption'])
    stdev = absorb.std_dev.reshape((3, 2))
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
    mean_list=mean.T.reshape((1, 6)).tolist()
    data_json['input']=mean_list[0]
    data_json['bean_num']=0.5
    modelinfo={'det_num_x': 10, 'det_num_y': 10, 'det_y': 0.03, 'det_x': 0.03, 'size_x': 0.5, 'size_x': 0.5, 'size_y': 0.5, 'med_margin': 0.0015}    #!20220119 constant!
    data_json['model_info']=modelinfo
    with open(folder1+file1,"w") as f:
        json.dump(data_json, f)

    plt.imshow(mean, interpolation='nearest', cmap="plasma")#'gist_gray')#"plasma")

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

def before_openmc(use_panels, sources_d_th, num_particles, seg_angles):
    batches = 100
    panel_density = 5.76
    src_E = None
    src_Str = 10
    num_sources = len(sources_d_th)
    energy_prob = (1)
    start = timeit.timeit()
    start_time = datetime.now()
    gen_materials_geometry_tallies(use_panels, panel_density)
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

def after_openmc(use_panels, sources_d_th, folder1, folder2, seg_angles, header):
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
    mm = process_aft_openmc(use_panels, folder1, file1, folder2, file2, sources, seg_angles, norm=True)
    return mm


def get_position_from_panelID(panel_id):

    """_summary_
    array
        d (2,0) | a (2,1)
        ________ _______
        e (1,0) | b (1,1)
        ________ _______
        f (0,0) | c (0,1)
    
    """

    if panel_id == 'a':
        id0, id1 = 2, 1
    elif panel_id == 'b':
        id0, id1 = 1, 1
    elif panel_id == 'c':
        id0, id1 = 0, 1
    elif panel_id == 'd':
        id0, id1 = 2, 0
    elif panel_id == 'e':
        id0, id1 = 1, 0
    elif panel_id == 'f':
        id0, id1 = 0, 0
    
    return id0, id1


def get_tetris_shape(shape_name):
    
    """_summary_
    array
        d | a
        _   _
        e | b
        _   _
        f | c
    
    """
    if shape_name == 'J':
        use_panels = ['a', 'd', 'e', 'f']
    elif shape_name == 'L':
        use_panels = ['c', 'd', 'e', 'f']
    elif shape_name == 'Z':
        use_panels = ['a', 'b', 'e', 'f']
    elif shape_name == 'T':
        use_panels = ['b', 'd', 'e', 'f']
    elif shape_name == 'S':
        use_panels = ['b', 'c', 'd', 'e']
    elif shape_name == 'full':
        use_panels = ['a', 'b', 'c', 'd', 'e', 'f']
    return use_panels


#%%

if __name__ == '__main__':
    num_sources = 1
    num_data = 1000
    seg_angles = 64
    dist_min = 50
    dist_max = 500
    source_energies = [0.5e6, 0.5e6]
    num_particles = 20000
    shape_list = ['S', 'J', 'T', 'Z', 'L']
    header = 'data'
    
    for shape_name in shape_list:

        folder1=f'openmc/data_tetris{shape_name}_{num_sources}src_{seg_angles}_data_20220821_v1.1/'
        folder2=f'openmc/data_tetris{shape_name}_{num_sources}src_{seg_angles}_fig_20220821_v1.1/'
        use_panels = get_tetris_shape(shape_name)

        for i in range(num_data):
            sources_d_th = [[np.random.randint(dist_min, dist_max), float(np.random.randint(0, 360) + np.random.random(1)), source_energies[i]] for i in range(num_sources)]
            for i in range(num_sources):
                print(f"dist {i}: " + str(sources_d_th[i][0]))
                print(f"angle {i}: " + str(sources_d_th[i][1]))
                print(f"energy {i}: " + str(sources_d_th[i][2]))
            before_openmc(use_panels, sources_d_th, num_particles, seg_angles)
            run_openmc()
            mm = after_openmc(use_panels, sources_d_th, folder1, folder2, seg_angles, header)

# %%
