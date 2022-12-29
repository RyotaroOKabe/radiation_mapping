#%%
"""
Created on 2022/12/27
@author: R.Okabe
"""

import glob
import matplotlib.pyplot as plt
import os
from utils.dataset import gen_materials, get_sources, gen_settings, output_process, run_openmc
import openmc
digits = 10

def gen_materials_geometry_tallies(a_num, panel_density):
    panel, insulator, outer = gen_materials(panel_density)  #! remove outer

    min_x = openmc.XPlane(x0=-100000, boundary_type='transmission')
    max_x = openmc.XPlane(x0=+100000, boundary_type='transmission')
    min_y = openmc.YPlane(y0=-100000, boundary_type='transmission')
    max_y = openmc.YPlane(y0=+100000, boundary_type='transmission')
    min_xx = openmc.XPlane(x0=-100100, boundary_type='vacuum')  #?
    max_xx = openmc.XPlane(x0=+100100, boundary_type='vacuum')
    min_yy = openmc.YPlane(y0=-100100, boundary_type='vacuum')
    max_yy = openmc.YPlane(y0=+100100, boundary_type='vacuum')    
    #for S1 layer
    min_x1 = openmc.XPlane(x0=-0.4, boundary_type='transmission')
    max_x1 = openmc.XPlane(x0=+0.4, boundary_type='transmission')
    min_y1 = openmc.YPlane(y0=-0.4, boundary_type='transmission')
    max_y1 = openmc.YPlane(y0=+0.4, boundary_type='transmission')
    #for S2 layer
    min_x2 = openmc.XPlane(x0=-0.5, boundary_type='transmission')
    max_x2 = openmc.XPlane(x0=+0.5, boundary_type='transmission')
    min_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')
    max_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')
    #for S3 layer
    min_x3 = openmc.XPlane(x0=-a_num/2, boundary_type='transmission')
    max_x3 = openmc.XPlane(x0=+a_num/2, boundary_type='transmission')
    min_y3 = openmc.YPlane(y0=-a_num/2, boundary_type='transmission')
    max_y3 = openmc.YPlane(y0=+a_num/2, boundary_type='transmission')

    s1_region = +min_x1 & -max_x1 & +min_y1 & -max_y1
    s2_region = +min_x2 & -max_x2 & +min_y2 & -max_y2
    s3_region = +min_x3 & -max_x3 & +min_y3 & -max_y3
    s4_region = +min_x & -max_x & +min_y & -max_y
    s5_region = +min_xx & -max_xx & +min_yy & -max_yy
    s1_cell = openmc.Cell(name='s1 cell', fill=panel, region=s1_region)
    s2_cell = openmc.Cell(name='s2 cell', fill=insulator, region= ~s1_region & s2_region)
    # Create a Universe to encapsulate a fuel pin
    cell_universe = openmc.Universe(name='universe', cells=[s1_cell, s2_cell])
    # Create fuel assembly Lattice
    assembly = openmc.RectLattice(name='detector arrays')
    assembly.pitch = (1, 1)
    assembly.lower_left = [-1 * a_num / 2.0] * 2
    assembly.universes = [[cell_universe] * a_num] * a_num
    # Create root Cell
    arrays_cell = openmc.Cell(name='arrays cell', fill=assembly, region = s3_region)
    root_cell = openmc.Cell(name='root cell', fill=None, region = ~s3_region & s4_region)
    outer_cell = openmc.Cell(name='outer cell', fill=None, region = ~s4_region & s5_region)  #?
    root_universe = openmc.Universe(name='root universe')
    root_universe.add_cell(arrays_cell)
    root_universe.add_cell(root_cell)
    root_universe.add_cell(outer_cell)
    root_universe.plot(width=(22, 22), basis='xy')
    # plt.show()
    # plt.savefig('save_fig/geometry.png')
    # plt.savefig('save_fig/geometry.pdf')
    # plt.close()
    # Create Geometry and export to "geometry.xml"
    geometry = openmc.Geometry(root_universe)
    geometry.export_to_xml()
    # Instantiate an empty Tallies object
    tallies = openmc.Tallies()
    # Instantiate a tally Mesh
    mesh = openmc.RegularMesh(mesh_id=1)
    mesh.dimension = [a_num, a_num]
    mesh.lower_left = [-a_num/2, -a_num/2]
    mesh.width = [1, 1]
    mesh_filter = openmc.MeshFilter(mesh)
    # Instantiate the Tally
    tally = openmc.Tally(name='mesh tally')
    tally.filters = [mesh_filter]
    tally.scores = ["absorption"]
    # Add mesh and Tally to Tallies
    tallies.append(tally)  #! check some com,mented out fucntions related to tallies. 
    tallies.export_to_xml()
    # Remove old HDF5 (summary, statepoint) files
    os.system('rm statepoint.*')
    os.system('rm summary.*')

def process_aft_openmc(a_num, folder, file, sources, seg_angles, norm, savefig=False):
    statepoints = glob.glob('statepoint.*.h5')
    sp = openmc.StatePoint(statepoints[-1])
    tally = sp.get_tally(name='mesh tally')
    # data = tally.get_values()
    df = tally.get_pandas_dataframe(nuclides=False)
    #? pd.options.display.float_format = '{:.2e}'.format
    fiss = df[df['score'] == 'absorption']
    mean = fiss['mean'].values.reshape((a_num, a_num))
    mean = output_process(mean, digits, folder, file, sources, seg_angles, norm, savefig)
    return mean

def before_openmc(a_num, sources_d_th, num_particles):
    batches = 100
    panel_density = 5.76
    src_E = None
    src_Str = 10
    energy_prob = (1)
    gen_materials_geometry_tallies(a_num, panel_density)
    sources = get_sources(sources_d_th)
    gen_settings(src_energy=src_E, src_strength=src_Str, en_prob=energy_prob, num_particles=num_particles, batch_size=batches, sources=sources) 

def after_openmc(a_num, sources_d_th, folder, seg_angles, header, record=None, savefig=False):
    num_sources = len(sources_d_th)
    d_a_seq = ""
    for i in range(num_sources):
        d_a_seq += '_' + str(round(sources_d_th[i][0], 5)) + '_' + str(round(sources_d_th[i][1], 5))
    # file1=header + d_a_seq + '.json'
    # file2=header + d_a_seq + '.png'
    file =header + d_a_seq
    isExist1 = os.path.exists(folder)
    if not isExist1:
        os.makedirs(folder)
        print("The new directory "+ folder +" is created!")
    if record is not None:
        with open(f'{folder}/record.txt', 'w') as f:
            for line in record:
                f.write(line + "\n")
    if savefig:
        folder2 = folder + '_fig'
        isExist2 = os.path.exists(folder2)
        if not isExist2:
            os.makedirs(folder2)
            print("The new directory "+ folder2 +" is created!")
    sources=get_sources(sources_d_th)
    mm = process_aft_openmc(a_num, folder, file, sources, seg_angles, norm=True, savefig=savefig)
    return mm

# %%
