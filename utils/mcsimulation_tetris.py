#%%
import glob
import matplotlib.pyplot as plt
import os
from utils.dataset import gen_materials, get_sources, gen_settings, output_process
import openmc
digits = 10

def gen_materials_geometry_tallies(use_panels, panel_density):
    panel, insulator, outer = gen_materials(panel_density)

    min_x = openmc.XPlane(x0=-100000, boundary_type='transmission')
    max_x = openmc.XPlane(x0=+100000, boundary_type='transmission')
    min_y = openmc.YPlane(y0=-100000, boundary_type='transmission')
    max_y = openmc.YPlane(y0=+100000, boundary_type='transmission')
    min_xx = openmc.XPlane(x0=-100100, boundary_type='vacuum')
    max_xx = openmc.XPlane(x0=+100100, boundary_type='vacuum')
    min_yy = openmc.YPlane(y0=-100100, boundary_type='vacuum')
    max_yy = openmc.YPlane(y0=+100100, boundary_type='vacuum')
    # for A inner layer
    Amin_x1 = openmc.XPlane(x0=+0.1, boundary_type='transmission')
    Amax_x1 = openmc.XPlane(x0=+0.9, boundary_type='transmission')
    Amin_y1 = openmc.YPlane(y0=+0.6, boundary_type='transmission')
    Amax_y1 = openmc.YPlane(y0=+1.4, boundary_type='transmission')
    #for A outer layer
    Amin_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')
    Amax_x2 = openmc.XPlane(x0=+1.0, boundary_type='transmission')
    Amin_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')
    Amax_y2 = openmc.YPlane(y0=+1.5, boundary_type='transmission')
    # for B inner layer (y-1 from A)
    Bmin_x1 = openmc.XPlane(x0=+0.1, boundary_type='transmission')
    Bmax_x1 = openmc.XPlane(x0=+0.9, boundary_type='transmission')
    Bmin_y1 = openmc.YPlane(y0=-0.4, boundary_type='transmission')
    Bmax_y1 = openmc.YPlane(y0=+0.4, boundary_type='transmission')
    #for B outer layer (y-1 from A)
    Bmin_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')
    Bmax_x2 = openmc.XPlane(x0=+1.0, boundary_type='transmission')
    Bmin_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')
    Bmax_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')
    # for C inner layer (y-2 from A)
    Cmin_x1 = openmc.XPlane(x0=+0.1, boundary_type='transmission')
    Cmax_x1 = openmc.XPlane(x0=+0.9, boundary_type='transmission')
    Cmin_y1 = openmc.YPlane(y0=-1.4, boundary_type='transmission')
    Cmax_y1 = openmc.YPlane(y0=-0.6, boundary_type='transmission')
    #for C outer layer (y-2 from A)
    Cmin_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')
    Cmax_x2 = openmc.XPlane(x0=+1.0, boundary_type='transmission')
    Cmin_y2 = openmc.YPlane(y0=-1.5, boundary_type='transmission')
    Cmax_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')
    # for D inner layer (x-1 from A)
    Dmin_x1 = openmc.XPlane(x0=-0.9, boundary_type='transmission')
    Dmax_x1 = openmc.XPlane(x0=-0.1, boundary_type='transmission')
    Dmin_y1 = openmc.YPlane(y0=+0.6, boundary_type='transmission')
    Dmax_y1 = openmc.YPlane(y0=+1.4, boundary_type='transmission')
    #for D outer layer (x-1 from A)
    Dmin_x2 = openmc.XPlane(x0=-1.0, boundary_type='transmission')
    Dmax_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')
    Dmin_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')
    Dmax_y2 = openmc.YPlane(y0=+1.5, boundary_type='transmission')
    # for E inner layer (x-1, y-1 from A)
    Emin_x1 = openmc.XPlane(x0=-0.9, boundary_type='transmission')
    Emax_x1 = openmc.XPlane(x0=+0.9, boundary_type='transmission')
    Emin_y1 = openmc.YPlane(y0=-0.4, boundary_type='transmission')
    Emax_y1 = openmc.YPlane(y0=+0.4, boundary_type='transmission')
    #for E outer layer (x-1, y-1 from A)
    Emin_x2 = openmc.XPlane(x0=-1.0, boundary_type='transmission')
    Emax_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')
    Emin_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')
    Emax_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')
    # for E inner layer (x-1, y-2 from A)
    Fmin_x1 = openmc.XPlane(x0=-0.9, boundary_type='transmission')
    Fmax_x1 = openmc.XPlane(x0=+0.9, boundary_type='transmission')
    Fmin_y1 = openmc.YPlane(y0=-1.4, boundary_type='transmission')
    Fmax_y1 = openmc.YPlane(y0=-0.6, boundary_type='transmission')
    #for F outer layer (x-1, y-2 from A)
    Fmin_x2 = openmc.XPlane(x0=-1.0, boundary_type='transmission')
    Fmax_x2 = openmc.XPlane(x0=+0.0, boundary_type='transmission')
    Fmin_y2 = openmc.YPlane(y0=-1.5, boundary_type='transmission')
    Fmax_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')
    #for S3 layer
    min_x3 = openmc.XPlane(x0=-1.0, boundary_type='transmission')
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
    # root_universe.plot(width=(20, 20), basis='xy')
    # plt.show()
    # plt.savefig(f'save_fig/geometry_tetris_{s}_v1.png')
    # plt.savefig(f'save_fig/geometry_tetris_{s}_v1.png')
    # plt.close()
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

def process_aft_openmc(use_panels, folder, file, sources, seg_angles, norm, savefig=False):
    statepoints = glob.glob('statepoint.*.h5')
    sp = openmc.StatePoint(statepoints[-1])
    tally = sp.get_tally(name='mesh tally')
    # data = tally.get_values()
    df = tally.get_pandas_dataframe(nuclides=False)
    #? pd.options.display.float_format = '{:.2e}'.format
    fiss = df[df['score'] == 'absorption']
    mean = fiss['mean'].values.reshape((3, 2))
    remove_panels = ['a', 'b', 'c', 'd', 'e', 'f']
    for mark in use_panels:
        remove_panels.remove(mark)
    # print(remove_panels)
    for mark in remove_panels:
        id0, id1 = get_position_from_panelID(mark)
        mean[id0, id1] = 0
    mean = output_process(mean, digits, folder, file, sources, seg_angles, norm, savefig)
    return mean

def before_openmc(use_panels, sources_d_th, num_particles):
    batches = 100
    panel_density = 5.76
    src_E = None
    src_Str = 10
    energy_prob = (1)
    gen_materials_geometry_tallies(use_panels, panel_density)
    sources = get_sources(sources_d_th)
    gen_settings(src_energy=src_E, src_strength=src_Str, en_prob=energy_prob, num_particles=num_particles, batch_size=batches, sources=sources) 

def after_openmc(use_panels, sources_d_th, folder, seg_angles, header, record=None, savefig=False):
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
        with open(f'{folder}/a_record.txt', 'w') as f:
            for line in record:
                f.write(line + "\n")
    if savefig:
        folder2 = folder + '_fig'
        isExist2 = os.path.exists(folder2)
        if not isExist2:
            os.makedirs(folder2)
            print("The new directory "+ folder2 +" is created!")
    sources=get_sources(sources_d_th)
    mm = process_aft_openmc(use_panels, folder, file, sources, seg_angles, norm=True, savefig=savefig)
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

# %%
