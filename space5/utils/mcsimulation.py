#%%
import glob
import matplotlib.pyplot as plt
import os
from utils.dataset import gen_materials, get_sources, gen_settings, output_process
import openmc
digits = 10

def gen_materials_geometry_tallies(*args, **kwargs):
    if len(args) == 2:
        # Function signature: gen_materials_geometry_tallies(a_num, panel_density)
        a_num, panel_density = args
        panel, insulator, outer = gen_materials(panel_density)

        # Rest of the function implementation for function1

        # Create Geometry and export to "geometry.xml"

        # Instantiate an empty Tallies object

        # Instantiate a tally Mesh

        # Instantiate the Tally

        # Add mesh and Tally to Tallies

        # Export to "tallies.xml"

        # Remove old HDF5 (summary, statepoint) files

        return  # Return the desired result
    elif len(args) == 2:
        # Function signature: gen_materials_geometry_tallies(use_panels, panel_density)
        use_panels, panel_density = args
        panel, insulator, outer = gen_materials(panel_density)

        # Rest of the function implementation for function2

        # Create Geometry and export to "geometry.xml"

        # Instantiate an empty Tallies object

        # Instantiate a tally Mesh

        # Instantiate the Tally

        # Add mesh and Tally to Tallies

        # Export to "tallies.xml"

        # Remove old HDF5 (summary, statepoint) files

        return  # Return the desired result
    else:
        raise ValueError("Invalid number of arguments.")


def process_aft_openmc(*args, **kwargs):
    if len(args) == 6:
        # Function signature: process_aft_openmc(a_num, folder, file, sources, seg_angles, norm, savefig=False)
        a_num, folder, file, sources, seg_angles, norm = args
        savefig = kwargs.get('savefig', False)
        statepoints = glob.glob('statepoint.*.h5')
        sp = openmc.StatePoint(statepoints[-1])
        tally = sp.get_tally(name='mesh tally')
        df = tally.get_pandas_dataframe(nuclides=False)
        fiss = df[df['score'] == 'absorption']
        mean = fiss['mean'].values.reshape((a_num, a_num))
        mean = output_process(mean, digits, folder, file, sources, seg_angles, norm, savefig)
        return mean
    elif len(args) == 7:
        # Function signature: process_aft_openmc(use_panels, folder, file, sources, seg_angles, norm, savefig=False)
        use_panels, folder, file, sources, seg_angles, norm = args
        savefig = kwargs.get('savefig', False)
        statepoints = glob.glob('statepoint.*.h5')
        sp = openmc.StatePoint(statepoints[-1])
        tally = sp.get_tally(name='mesh tally')
        df = tally.get_pandas_dataframe(nuclides=False)
        fiss = df[df['score'] == 'absorption']
        mean = fiss['mean'].values.reshape((3, 2))
        remove_panels = ['a', 'b', 'c', 'd', 'e', 'f']
        for mark in use_panels:
            remove_panels.remove(mark)
        for mark in remove_panels:
            id0, id1 = get_position_from_panelID(mark)
            mean[id0, id1] = 0
        mean = output_process(mean, digits, folder, file, sources, seg_angles, norm, savefig)
        return mean
    else:
        raise ValueError("Invalid number of arguments.")

def before_openmc(*args, **kwargs):
    if len(args) == 3:
        # Function signature: before_openmc(a_num, sources_d_th, num_particles)
        a_num, sources_d_th, num_particles = args
        batches = 100
        panel_density = 5.76
        src_E = None
        src_Str = 10
        energy_prob = (1)
        gen_materials_geometry_tallies(a_num, panel_density)
        sources = get_sources(sources_d_th)
        gen_settings(src_energy=src_E, src_strength=src_Str, en_prob=energy_prob, num_particles=num_particles, batch_size=batches, sources=sources) 
    elif len(args) == 4:
        # Function signature: before_openmc(use_panels, sources_d_th, num_particles)
        use_panels, sources_d_th, num_particles = args
        batches = 100
        panel_density = 5.76
        src_E = None
        src_Str = 10
        energy_prob = (1)
        gen_materials_geometry_tallies(use_panels, panel_density)
        sources = get_sources(sources_d_th)
        gen_settings(src_energy=src_E, src_strength=src_Str, en_prob=energy_prob, num_particles=num_particles, batch_size=batches, sources=sources) 
    else:
        raise ValueError("Invalid number of arguments.")


def after_openmc(*args, **kwargs):
    if len(args) == 7:
        # Function signature: after_openmc(a_num, sources_d_th, folder, seg_angles, header, record=None, savefig=False)
        a_num, sources_d_th, folder, seg_angles, header, record, savefig = args
        num_sources = len(sources_d_th)
        d_a_seq = ""
        for i in range(num_sources):
            d_a_seq += '_' + str(round(sources_d_th[i][0], 5)) + '_' + str(round(sources_d_th[i][1], 5))
        file = header + d_a_seq
        isExist1 = os.path.exists(folder)
        if not isExist1:
            os.makedirs(folder)
            print("The new directory " + folder + " is created!")
        if record is not None:
            with open(f'{folder}/a_record.txt', 'w') as f:
                for line in record:
                    f.write(line + "\n")
        if savefig:
            folder2 = folder + '_fig'
            isExist2 = os.path.exists(folder2)
            if not isExist2:
                os.makedirs(folder2)
                print("The new directory " + folder2 + " is created!")
        sources = get_sources(sources_d_th)
        mm = process_aft_openmc(a_num, folder, file, sources, seg_angles, norm=True, savefig=savefig)
        return mm
    elif len(args) == 8:
        # Function signature: after_openmc(use_panels, sources_d_th, folder, seg_angles, header, record=None, savefig=False)
        use_panels, sources_d_th, folder, seg_angles, header, record, savefig = args
        num_sources = len(sources_d_th)
        d_a_seq = ""
        for i in range(num_sources):
            d_a_seq += '_' + str(round(sources_d_th[i][0], 5)) + '_' + str(round(sources_d_th[i][1], 5))
        file = header + d_a_seq
        isExist1 = os.path.exists(folder)
        if not isExist1:
            os.makedirs(folder)
            print("The new directory " + folder + " is created!")
        if record is not None:
            with open(f'{folder}/a_record.txt', 'w') as f:
                for line in record:
                    f.write(line + "\n")
        if savefig:
            folder2 = folder + '_fig'
            isExist2 = os.path.exists(folder2)
            if not isExist2:
                os.makedirs(folder2)
                print("The new directory " + folder2 + " is created!")
        sources = get_sources(sources_d_th)
        mm = process_aft_openmc(use_panels, folder, file, sources, seg_angles, norm=True, savefig=savefig)
        return mm
    else:
        raise ValueError("Invalid number of arguments.")


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


