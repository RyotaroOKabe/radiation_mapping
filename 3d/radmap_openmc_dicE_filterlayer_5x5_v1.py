from contextlib import redirect_stderr
import glob
import imp
from IPython.display import Image
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.figure import Figure    #!20220210
import scipy.stats
import numpy as np
import pandas as pd
import os
import json #!20220119
#import torch   #!20220125 20220430 remove for its incompatibility
import time, timeit   #!20220224
from datetime import datetime   #!20220224
import openmc
#from dataset import *


GPU_INDEX = 0
USE_CPU = False
# print torch.cuda.is_available()
#if torch.cuda.is_available() and not USE_CPU:
#    DEFAULT_DEVICE = torch.device("cuda:%d"%GPU_INDEX) 
#    torch.cuda.set_device(GPU_INDEX)
#    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
#else:
#    DEFAULT_DEVICE = torch.device("cpu")

###=================Input parameter======================

rad_source_x= [50, 7]

###=======================================

#def gen_materials_geometry_tallies():
def gen_materials_geometry_tallies(panel_density, e_filter, *energy):
    # 1.6 enriched fuel
    panel = openmc.Material(name='CdZnTe')  #!20220302
    #panel.set_density('g/cm3', 5.8)
    panel.set_density('g/cm3', panel_density)#5.8)
    #panel.add_nuclide('U235', 0.33)
    #panel.add_nuclide('U238', 0.33)
    panel.add_nuclide('Cd114', 33, percent_type='ao')
    panel.add_nuclide('Zn64', 33, percent_type='ao')
    panel.add_nuclide('Te130', 33, percent_type='ao')

    # zircaloy
    insulator = openmc.Material(name='Zn')  #!20220302
    insulator.set_density('g/cm3', 1)
    #zink.add_nuclide('Zn64', 1)
    insulator.add_nuclide('Pb208', 11.35)
    
    outer = openmc.Material(name='Outer_CdZnTe')
    outer.set_density('g/cm3', panel_density)#5.8)
    outer.add_nuclide('Cd114', 33, percent_type='ao')
    outer.add_nuclide('Zn64', 33, percent_type='ao')
    outer.add_nuclide('Te130', 33, percent_type='ao')

    materials = openmc.Materials(materials=[panel, insulator, outer])
    #materials.cross_sections = '/home/rokabe/data1/openmc/endfb71_hdf5/cross_sections.xml'
    materials.export_to_xml()

    #os.system("cat materials.xml")

    #for root cell
    #min_x = openmc.XPlane(x0=-100000, boundary_type='vacuum')
    #max_x = openmc.XPlane(x0=+100000, boundary_type='vacuum')
    #min_y = openmc.YPlane(y0=-100000, boundary_type='vacuum')
    #max_y = openmc.YPlane(y0=+100000, boundary_type='vacuum')

    min_x = openmc.XPlane(x0=-100000, boundary_type='transmission')
    max_x = openmc.XPlane(x0=+100000, boundary_type='transmission')
    min_y = openmc.YPlane(y0=-100000, boundary_type='transmission')
    max_y = openmc.YPlane(y0=+100000, boundary_type='transmission')

    #for S1 layer
    min_x1 = openmc.XPlane(x0=-0.4, boundary_type='transmission')   #!20220301
    max_x1 = openmc.XPlane(x0=+0.4, boundary_type='transmission')
    min_y1 = openmc.YPlane(y0=-0.4, boundary_type='transmission')
    max_y1 = openmc.YPlane(y0=+0.4, boundary_type='transmission')

    #for S2 layer
    min_x2 = openmc.XPlane(x0=-0.5, boundary_type='transmission')   #!20220124
    max_x2 = openmc.XPlane(x0=+0.5, boundary_type='transmission')
    min_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')
    max_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')

    #for S3 layer
    min_x3 = openmc.XPlane(x0=-2.5, boundary_type='transmission')     #!20220124
    max_x3 = openmc.XPlane(x0=+2.5, boundary_type='transmission')
    min_y3 = openmc.YPlane(y0=-2.5, boundary_type='transmission')
    max_y3 = openmc.YPlane(y0=+2.5, boundary_type='transmission')
    
    #for outer insulator cell
    #min_xx = openmc.XPlane(x0=-100100, boundary_type='vacuum')
    #max_xx = openmc.XPlane(x0=+100100, boundary_type='vacuum')
    #min_yy = openmc.YPlane(y0=-100100, boundary_type='vacuum')
    #max_yy = openmc.YPlane(y0=+100100, boundary_type='vacuum')
    
    min_xx = openmc.XPlane(x0=-100100, boundary_type='vacuum')
    max_xx = openmc.XPlane(x0=+100100, boundary_type='vacuum')
    min_yy = openmc.YPlane(y0=-100100, boundary_type='vacuum')
    max_yy = openmc.YPlane(y0=+100100, boundary_type='vacuum')    

    #s1 region
    s1_region = +min_x1 & -max_x1 & +min_y1 & -max_y1

    #s2 region
    s2_region = +min_x2 & -max_x2 & +min_y2 & -max_y2

    #s3 region
    s3_region = +min_x3 & -max_x3 & +min_y3 & -max_y3

    #s4 region
    s4_region = +min_x & -max_x & +min_y & -max_y
    
    #s5 region
    s5_region = +min_xx & -max_xx & +min_yy & -max_yy

    #define s1 cell
    s1_cell = openmc.Cell(name='s1 cell', fill=panel, region=s1_region)

    #define s2 cell
    s2_cell = openmc.Cell(name='s2 cell', fill=insulator, region= ~s1_region & s2_region)

    # Create a Universe to encapsulate a fuel pin
    cell_universe = openmc.Universe(name='universe', cells=[s1_cell, s2_cell])   #!20220117

    # Create fuel assembly Lattice
    assembly = openmc.RectLattice(name='detector arrays')
    assembly.pitch = (1, 1) #(1, 1)   #!20220124
    assembly.lower_left = [-1 * 5 / 2.0] * 2    #!20220626
    assembly.universes = [[cell_universe] * 5] * 5

    #print(assembly)

    # Create root Cell
    arrays_cell = openmc.Cell(name='arrays cell', fill=assembly, region = s3_region)
    root_cell = openmc.Cell(name='root cell', fill=None, region = ~s3_region & s4_region)   #!20220117
    #outer_cell = openmc.Cell(name='outer cell', fill=outer, region = ~s4_region & s5_region)   #!20220124
    outer_cell = openmc.Cell(name='outer cell', fill=None, region = ~s4_region & s5_region)   #!20220124

    root_universe = openmc.Universe(name='root universe')
    root_universe.add_cell(arrays_cell)
    root_universe.add_cell(root_cell)
    root_universe.add_cell(outer_cell)

    root_universe.plot(width=(22, 22), basis='xy')     #!20220124
    plt.show()   #!20220117
    plt.savefig('savefig/geometry_20220201.png')   #!20220117
    plt.close()

    # Create Geometry and export to "geometry.xml"
    geometry = openmc.Geometry(root_universe)
    geometry.export_to_xml()

    #os.system("cat geometry.xml")


    #def gen_tallies():

    # Instantiate an empty Tallies object
    tallies = openmc.Tallies()

    # Instantiate a tally Mesh
    mesh = openmc.RegularMesh(mesh_id=1)
    mesh.dimension = [5, 5]
    mesh.lower_left = [-2.5, -2.5]  #[-10, -10]     #!20220124
    mesh.width = [1, 1] #[2, 2]  #!20220124

    # Instantiate tally Filter
    mesh_filter = openmc.MeshFilter(mesh)

    # Instantiate energy Filter
    #energy_filter = openmc.EnergyFilter([0, 0.625, 20.0e6])
    # Instantiate the Tally
    tally = openmc.Tally(name='mesh tally')
    
    if e_filter:
        energy_filter = openmc.EnergyFilter(*energy)    #!20220204
        tally.filters = [mesh_filter, energy_filter]    #!20220204
    
    else:
        tally.filters = [mesh_filter]    #!20220204
        
    #tally.filters = [mesh_filter]    #! 20220201 / #!20220117 Test!
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

    # Add mesh and tally to Tallies
    #tallies.append(tally)

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

    # Add mesh and tally to Tallies
    #tallies.append(tally)

    # Export to "tallies.xml"
    tallies.export_to_xml()

    #os.system("cat tallies.xml")

    # Remove old HDF5 (summary, statepoint) files
    os.system('rm statepoint.*')
    os.system('rm summary.*')


#def gen_settings(rad_source1=rad_source_x):
def gen_settings(src_energy=None, src_strength=1, en_source=1e6, en_prob=1, num_particles=10000, batch_size=100, source_x=rad_source_x[0], source_y=rad_source_x[1]): #!20220224
    # Create a point source
    #point = openmc.stats.Point((2, 13, 0))
    #source = openmc.Source(space=point)
    #point1 = openmc.stats.Point((30, 13, 0))
    #point1 = openmc.stats.Point((rad_source1[0], rad_source1[1], 0))
    point1 = openmc.stats.Point((source_x, source_y, 0))
    #source1 = openmc.Source(space=point1, particle='photon')  #!20220118
    source1 = openmc.Source(space=point1, particle='photon', energy=src_energy, strength=src_strength)  #!20220204    #!20220118
    #point2 = openmc.stats.Point((-50, 6, 0))
    #source2 = openmc.Source(space=point2, particle='photon')  #!20220118
    #point3 = openmc.stats.Point((1, -20, 0))
    #source3 = openmc.Source(space=point3, particle='photon')  #!20220118
    #source.particle = 'photon'  #!20220117

    #!==================== 20220223
    #source1.energy = openmc.stats.Uniform(a=en_a, b=en_b)
    source1.energy = openmc.stats.Discrete(x=en_source, p=en_prob)
    #!====================

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'  #!20220118
    settings.photon_transport = True  #!20220117
    #settings.electron_treatment = 'led' #!20220117
    #settings.source = source
    settings.source = [source1] #, source2, source3]     #!20220118
    settings.batches = batch_size #100
    settings.inactive = 10
    settings.particles = num_particles

    settings.export_to_xml()

    #os.system("cat settings.xml")


def run_openmc():
    # Run OpenMC!
    openmc.run()


def process_aft_openmc(folder1='random_savearray/', file1='detector_1source_20220118.txt', \
                        folder2='random_savefig/', file2='detector_1source_20220118.png',\
                            source_x=100, source_y=100, norm=True):
    # We do not know how many batches were needed to satisfy the
    # tally trigger(s), so find the statepoint file(s)
    statepoints = glob.glob('statepoint.*.h5')

    # Load the last statepoint file
    sp = openmc.StatePoint(statepoints[-1])

    # Find the mesh tally with the StatePoint API
    tally = sp.get_tally(name='mesh tally')

    # Print a little info about the mesh tally to the screen
    #print(tally)

    #print("tally.sum")  #!20220210
    #print(tally.sum)
    #print(tally.sum.shape)


    # Get the relative error for the thermal fission reaction
    # rates in the four corner pins
    data = tally.get_values()#scores=['absorption'])
    #print(data)
    #print(data.shape)

    # Get the relative error for the thermal fission reaction
    # rates in the four corner pins
    #data = tally.get_values(scores=['absorption'],     #!20220118  Test!!
                            #filters=[openmc.MeshFilter, openmc.EnergyFilter], \
                            #filter_bins=[((1,1),(1,10), (10,1), (10,10)), \
                                        #((0., 0.625),)], value='rel_err')
    #print(data)     #!20220118  Test!!

    # Get a pandas dataframe for the mesh tally data
    df = tally.get_pandas_dataframe(nuclides=False)

    # Set the Pandas float display settings
    pd.options.display.float_format = '{:.2e}'.format

    # Print the first twenty rows in the dataframe
    df#.head(20)

    # Extract thermal absorption rates from pandas
    fiss = df[df['score'] == 'absorption']
    #fiss = fiss[fiss['energy low [eV]'] == 0.0]
    #fiss = fiss[fiss['energy low [eV]'] != 0.0]     #!20220118 Test!

    # Extract mean and reshape as 2D NumPy arrays
    mean = fiss['mean'].values.reshape((5,5)) # numpy array   #!20220118
    mean = np.transpose(mean)   #!20220502 Adjust the incorrect axis setting!
    max = mean.max()        #!20220205
    if norm:    #!20220201
        #max = mean.max()       #!20220205
        mean_me = mean.mean()   #!20220227
        mean_st = mean.std()
        mean = (mean-mean_me)/mean_st
        

    #print(mean)
    #print(type(mean))
    #print(mean.shape)
    #folder = 'random_savearray/'
    #file = 'detector_1source_20220118.txt'
    #np.savetxt(fname=folder1+file1, X=mean, delimiter=' ')#, newline='\n', header='', footer='', comments='# ', encoding=None)

    #!20220210
    absorb = tally.get_slice(scores=['absorption'])
    #print("absorption")
    #print(absorb)
    #print(type(absorb))
    #print(absorb.shape)
    #print("std_dev")
    stdev = absorb.std_dev.reshape((5,5))
    stdev_max = stdev.max()
    #print(stdev)

    #==================================
        
    data_json={} #!20220119
    data_json['source']=[source_x, source_y]
    #print('source: ' + str(type([source_x, source_y])))
    data_json['intensity']=100   #!20220119 tentative value!
    data_json['miu_detector']=0.3   #!20220119 constant!
    data_json['miu_medium']=1.2   #!20220119 constant!
    data_json['miu_air']=0.00018   #!20220119 constant!
    data_json['output']=get_output([source_x, source_y]).tolist()
    #print('output: ' + str(type(data_json['output'])))
    data_json['miu_de']=0.5   #!20220119 constant!
    mean_list=mean.T.reshape((1, 25)).tolist()
    #print('mean_list: ' + str(type(mean_list)))
    data_json['input']=mean_list[0]  #!20220119 Notice!!!
    data_json['bean_num']=0.5   #!20220119 constant!
    modelinfo={'det_num_x': 10, 'det_num_y': 10, 'det_y': 0.03, 'det_x': 0.03, 'size_x': 0.5, 'size_x': 0.5, 'size_y': 0.5, 'med_margin': 0.0015}    #!20220119 constant!
    data_json['model_info']=modelinfo   #!20220119 constant!
    #print(data_json)
    #print(type(data_json))
    # create json object from dictionary
    #json = json.dumps(data_json)
    # open file for writing, "w" 
    #f = open(folder1+file1,"w")
    # create json object from dictionary
    #json.dump(data_json, f)
    # write json object to file
    #f.write(json)
    # close file
    #f.close()
    with open(folder1+file1,"w") as f:
        json.dump(data_json, f)
    
    #==================================
    print("mean_max:")
    print(max)
    print("stdev_max:")
    print(stdev_max)
    print("mean/stdev ratio:")
    print(max/stdev_max)

    plt.imshow(mean, interpolation='nearest', cmap="plasma")       #!20220118
    #plt.title('absorption rate')
    #ds, ag = file2[:-5].split('_')
    ds, ag = file2[:-5].split('_')[1:]  #!20220517
    plt.title('dist: ' + ds + ',  angle: ' + ag + '\nMean_max: ' + str(max) + '\nStdev_max: ' + str(stdev_max))
    plt.xlabel('y')  #!20220502 Adjust the incorrect axis setting!
    plt.ylabel('x')  #!20220502 Adjust the incorrect axis setting!
    plt.colorbar()
    #plt.show()   #!20220117
    #plt.savefig('random_savefig/abs_rate_20220118_6.png')   #!20220117
    plt.savefig(folder2 + file2) #   'random_savefig/abs_rate_20220118_6.png')   #!20220117
    plt.close()


def get_output(source):
    sec_center=np.linspace(-np.pi,np.pi,41)
    output=np.zeros(40)
    sec_dis=2*np.pi/40.
    angle=np.arctan2(source[1],source[0])
    before_indx=int((angle+np.pi)/sec_dis)
    if before_indx>=40: #!20220430 (actually no need to add these two lines..)
        before_indx-=40
    after_indx=before_indx+1
    if after_indx>=40:
        after_indx-=40
    w1=abs(angle-sec_center[before_indx])
    w2=abs(angle-sec_center[after_indx])
    if w2>sec_dis:
        w2=abs(angle-(sec_center[after_indx]+2*np.pi))
        #print w2
    output[before_indx]+=w2/(w1+w2)
    output[after_indx]+=w1/(w1+w2)
    # print before_indx,output[before_indx],after_indx,output[after_indx],angle/np.pi*180
    # raw_input()
    return output

def before_openmc(rad_dist, rad_angle, num_particles):
#if __name__ == '__main__':
    ###=================Input parameter======================
    #num_data = 100
    batches = 100
    panel_density = 5.76 #g/cm3
    src_E = None    #[1,3]
    src_Str = 10
    #num_particles = 5000
    #dist_min = 100
    #dist_max = 1000
    #dist = 100
    #angle = 0
    #idx = 112
    #energy = [0, 0.625, 20.0e6]     #!20220128
    energy_filter_range = [0.1e6, 2e6]     #!20220223
    e_filter_tf=False
    source_energy = (0.5e6)
    energy_prob = (1)
    #energy = [7.5, 19]  
    ###=======================================
    start = timeit.timeit()
    start_time = datetime.now()
    
    gen_materials_geometry_tallies(panel_density, e_filter_tf, energy_filter_range)     #!20220205
    j=batches
    #for j in range(10,batches, 10):
    #for i in range(num_data):
    #rad_dist=dist   #np.random.randint(dist_min, dist_max) + np.random.random(1)    #!20220128
    #rad_dist=np.random.randint(dist_min, dist_max) + np.random.random(1)
        #rad_angle=angle  #np.random.randint(0, 359) + np.random.random(1)    #!20220128
    #rad_angle=np.random.randint(0, 359) + np.random.random(1)
    theta=rad_angle*np.pi/180
        #rad_source=[float(rad_dist*np.cos(theta)), float(rad_dist*np.sin(theta))]
    rad_x, rad_y=[float(rad_dist*np.cos(theta)), float(rad_dist*np.sin(theta))]   #!20220119
    print([rad_x, rad_y])
    get_output([rad_x, rad_y])
    
        #gen_settings(rad_sources1=rad_source)
    gen_settings(src_energy=src_E, src_strength=src_Str,  en_source=source_energy, en_prob=energy_prob, num_particles=num_particles, batch_size=j, source_x=rad_x, source_y=rad_y)    #!20220224
        #gen_tallies()
        
        #openmc.run()
#openmc.run(mpi_args=['mpiexec', '-n', '4'])
        #openmc.run(mpi_args=['mpiexec', '-n', '4', "-s", '11'])
        #openmc.run(threads=11)

#def after_openmc(rad_dist, rad_angle):      
def after_openmc(dist, angle, folder1, folder2, header):    #!20220517
        #folder1='random_savearray/'
        #file1=str(round(rad_dist[0], 5)) + '_' + str(round(rad_angle[0], 5)) + '.json'
        #file1=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '_' + str(idx) + '_' + str(j)+ '.json'
        #file1=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '_' + str(idx) + '_' + str(num_particles)+ '.json'
        #folder2='random_savefig/'
        #file2=str(round(rad_dist[0], 5)) + '_' + str(round(rad_angle[0], 5)) + '.png'
        #file2=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '_' + str(idx) + '_' + str(j)+ '.png'
        #file2=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '_' + str(idx) + '_' + str(num_particles)+ '.png'
    #folder1='filterpath/filterdata_v9/'
    #file1=str(round(rad_dist[0], 5)) + '_' + str(round(rad_angle[0], 5)) + '.json'
    file1=header + "_" + str(round(dist, 5)) + '_' + str(round(angle, 5)) + '.json'
    #folder2='filterpath/filterfig_v9/'
    #file2=str(round(rad_dist[0], 5)) + '_' + str(round(rad_angle[0], 5)) + '.png'
    file2=header + "_" + str(round(dist, 5)) + '_' + str(round(angle, 5)) + '.png'

    # Check whether the specified path exists or not
    isExist1 = os.path.exists(folder1)
    if not isExist1:
    # Create a new directory because it does not exist 
        os.makedirs(folder1)
        print("The new directory "+ folder1 +" is created!")
        
    # Check whether the specified path exists or not
    isExist2 = os.path.exists(folder2)
    if not isExist2:
    # Create a new directory because it does not exist 
        os.makedirs(folder2)
        print("The new directory "+ folder2 +" is created!")
    
    theta=angle*np.pi/180
    rad_x, rad_y=[float(dist*np.cos(theta)), float(dist*np.sin(theta))]   #!20220119
            
    process_aft_openmc(folder1, file1, folder2, file2, rad_x, rad_y, norm=True)  #!20220201 #!20220119
    
        #file11=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '_' + str(idx+1) + '_' + str(j)+ '.json'
        #file22=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '_' + str(idx+1) + '_' + str(j)+ '.png'
        #process_aft_openmc(folder1, file11, folder2, file22, rad_x, rad_y, norm=True)  #!20220201 #!20220119

    #end = time.time()
    #end_time = datetime.now()
    #print("Start at " + str(start_time))
    #print("Finish at " + str(end_time))
    #time_s = end - start
    #print("Total time [s]: " + str(time_s))
    #print(time.strftime('%H:%M:%S', time.gmtime(time_s)))


if __name__ == '__main__':
    num_data = 40
    #dist = 100
    #num_particles = 500000
    #header = 'near'
    #header_dist_dict = {'near': 30, 'far': 200}
    header_dist_particles_dict = {'near': [20, 10000], 'far': [200, 500000]}   #!20220518
    folder1='openmc/disc_filter_data_20220626_v1.1/'
    folder2='openmc/disc_filter_fig_20220626_v1.1/'
    #angle_list = [a*360/num_data for a in range(num_data)]
    #angle_list = [1+a*360/num_data - 180 for a in range(num_data)]
    #angle_list = [0.1+a*360/num_data for a in range(num_data)]
    angle_list = [0.1+a*360/num_data -180 for a in range(num_data)]
    print("angle_list for " + str(num_data) +" sections of angles:")
    print(angle_list)
    for header in header_dist_particles_dict.keys():
        dist = header_dist_particles_dict[header][0]
        num_particles = header_dist_particles_dict[header][1]
        for angle in angle_list:
            print("]]]]]]]]]]]]]]]]]")
            print("header: " + header)
            print("Distance: " + str(dist))
            print("angle: " + str(angle))
            before_openmc(dist, angle, num_particles)
            openmc.run()
            #after_openmc(dist, angle)
            mm = after_openmc(dist, angle, folder1, folder2, header)
