#%%
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import pickle as pkl

#sys.path.append('../')
sys.path.append('./')   #!20220331
from train_torch_openmc_debug_20220508 import *
#from gen_openmc_uniform_v2 import *
#from gen_openmc_data_discrete_v1 import *

from cal_param_v1 import *

import matplotlib.pyplot as plt #!20220509
from matplotlib.figure import Figure   
from matplotlib.patches import Wedge
import imageio  #!20220520

#record_data=False
record_data=True

#============================= #!20220331
#GPU_INDEX = 2#0
#USE_CPU = False
# print torch.cuda.is_available()
#if torch.cuda.is_available() and not USE_CPU:
#    DEFAULT_DEVICE = torch.device("cuda:%d"%GPU_INDEX) 
#    torch.cuda.set_device(GPU_INDEX)
#    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
#else:
#    DEFAULT_DEVICE = torch.device("cpu")
#
#DEFAULT_DEVICE = torch.device("cuda")
#DEFAULT_DTYPE = torch.double
#============================= #!20220331

file_header = "A20220627_v4.2"

#recordpath = 'mapping_0803' #?pkl files with python2 is stored
recordpath = 'mapping_data/mapping_' + file_header
if __name__ == '__main__' and record_data:
    if not os.path.isdir(recordpath):
        os.mkdir(recordpath)
    os.system('rm ' + recordpath + "/*")    #!20220509

jsonpath = recordpath + "_json/"
if __name__ == '__maintribution__' and record_data:
    if not os.path.isdir(jsonpath):
        os.mkdir(jsonpath)
    os.system('rm ' + jsonpath + "*")    #!20220509

figurepath = recordpath + "_figure/"
if __name__ == '__main__' and record_data:
    if not os.path.isdir(figurepath):
        os.mkdir(figurepath)
    os.system('rm ' + figurepath + "*")    #!20220509
    
predictpath = recordpath + "_predicted/"
if __name__ == '__main__' and record_data:
    if not os.path.isdir(predictpath):
        os.mkdir(predictpath)
    os.system('rm ' + predictpath + "*")    #!20220509


#model_path = '../2source_unet_model.pt'    #!20220331
model_path = 'save_model/model_20220627_openmc_4cm_4x4_10000_ep500_bs256_dsc_new_test_v1.1_model.pt'
model =torch.load(model_path)


DT = 0.1  # time tick [s]
SIM_TIME = 70.0
STATE_SIZE = 3
LM_SIZE = 3
#RSID = np.array([[0.0,5.0,5000000],[10.0,5.0,10000000]])   # source location (in detector frame) and intensity
#RSID = np.array([0.0,5.0,100000000])   #1 source
#RSID = np.array([10.0,20.0,100000000])   #1 source
#RSID = np.array([0.0,10.0,100000000])   #1 source
#RSID = np.array([5.0,5.0,100000000])   #1 source
RSID = np.array([-7.0,10.5,100000000])   #1 source


#SIM_STEP=25
SIM_STEP=10  #!20220505

#%%
def pi_2_pi(angle): #change 0<=theta<2pi to -pi<=theta<pi (move the pi<=theta<2pi to -pi<=theta<0) 
    return (angle + math.pi) % (2 * math.pi) - math.pi

def calc_input(time):
    '''
    a simplified open loop controller to make the robot move in a circle trajectory
    Input: time
    Output: control input of the robot
    '''

    if time <= 0.:  # wait at first
        v = 0.0
        yawrate = 0.0
    else:
        #v = 1.0  # [m/s]
        v = 1.0  # [m/s]    #!20220509
        #yawrate = 0.1  # [rad/s]
        yawrate = 0.1  # [rad/s]

    u = np.array([v, yawrate]).reshape(2, 1)

    return u

#%%
def motion_model(x, u):
    '''
    a simplified motion model of the robot that the detector is installed on.
    Input: x is the state (pose) of the robot at previous step, u is the control input of the robot
    Output: x is the state of the robot for the next step
    '''

    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = np.dot(F, x) + np.dot(B, u)

    x[2, 0] = pi_2_pi(x[2, 0])
    return x

#%%
#def openmc_simulation_uniform(source):
def openmc_simulation_uniform(source, header):  #!20220509
    
    batches = 100
    panel_density = 5.76 #g/cm3
    src_E = None
    src_Str = 10
    num_particles = 100000
    energy_filter_range = [0.1e6, 2e6]
    e_filter_tf=False
    energy_a = 0.5e6
    energy_b = 1e6
    #rad_x= source[0]
    #rad_y= source[1]
    rad_x= source[0]*100    #!20220509 use [cm] instead of m in openmc
    rad_y= source[1]*100
    rad_dist = np.sqrt(rad_x**2 + rad_y**2) #!20220331 I need to change it later..
    rad_angle = (np.arccos(rad_x/rad_dist)*180/np.pi)%360
    
    #gen_materials_geometry_tallies(panel_density, e_filter_tf, energy_filter_range)
    #get_output([rad_x, rad_y])
    #gen_settings(src_energy=src_E, src_strength=src_Str,  en_a=energy_a, en_b=energy_b, num_particles=num_particles, batch_size=batches, source_x=rad_x, source_y=rad_y)
    before_openmc(rad_dist, rad_angle, num_particles)   #!20220508
    openmc.run()
    #file1=str(round(rad_dist, 4)) + '_' + str(rad_angle) + '.json'
    #file2=str(round(rad_dist, 4)) + '_' + str(rad_angle) + '.png'
    #mm = process_aft_openmc(jsonpath+"/", file1, figurepath+"/", file2, rad_x, rad_y, norm=True) #norm=True)
    #folder1=str(round(rad_dist, 4)) + '_' + str(rad_angle) + '.json'
    #folder2=str(round(rad_dist, 4)) + '_' + str(rad_angle) + '.png'
    #folder1='openmc/discrete_data_20220508_v1.1/'
    #folder2='openmc/discrete_fig_20220508_v1.1/'
    #mm = after_openmc(rad_dist, rad_angle, jsonpath, figurepath)    #!20220508
    mm = after_openmc(rad_dist, rad_angle, jsonpath, figurepath, header)    #!20220509
    return mm

#%%
def main():
    time=0
    #step=-1
    step=0
    

    relsrc = []   #!20220509
    xTrue_record = []
    d_record=[]
    angle_record=[]
    predout_record=[]
    
    #hxTrue_record = []
    #source_record = []
    
    xTrue = np.zeros((STATE_SIZE, 1))
    hxTrue = xTrue # pose (position and orientation) of the robot (detector) in world frame

    while SIM_TIME >= time:
        time += DT
        step+=1

        
        u = calc_input(time)

        xTrue = motion_model(xTrue, u)


        hxTrue = np.hstack((hxTrue, xTrue))

        det_output=None
        predict=None

        xTrue_record.append(xTrue)  #!20220509
        #hxTrue_record.append(hxTrue)  #!20220509

        if step%SIM_STEP==0:
            print('STEP %d'%step)
            #source_list=[] #!20220331
            source=[]
            #for i in range(RSID.shape[0]):
                # convert source location in world frame to detector frame
                #dx = RSID[i, 0] - xTrue[0, 0]
                #dy = RSID[i, 1] - xTrue[1, 0]
                #d = math.sqrt(dx**2 + dy**2)
                #angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])

                #x=d*np.cos(angle)
                #y=d*np.sin(angle)

                #rate=RSID[i,2]

                #source_list.append([x,y,rate])
            dx = RSID[0] - xTrue[0, 0]
            dy = RSID[1] - xTrue[1, 0]
            d = math.sqrt(dx**2 + dy**2)
            angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
            x=d*np.cos(angle)
            y=d*np.sin(angle)
            rate=RSID[2]
            #source.append(x,y,rate)
            source=[x,y,rate]   #relative vector
            
            relsrc.append([x,y])#!20220509
            d_record.append([step, d])
            angle_record.append([step, angle*180/np.pi])

            print(step,'simulation start')
            print("source")
            print(source)
            #print(len(source))
            
            # replace this line with openmc simulation function
            #det_output=simulation(source_list)     #!20220331
            #det_output=openmc_simulation_uniform(source)
            det_output=openmc_simulation_uniform(source, 'STEP%.3d'%step)
            #print('det_output')
            #print(det_output)
            '''
            The simulation function simulate the detector response given the radiation source position and intensity.
            The input of this function is a list of sources location (in detector frame, (0,0) is the center of the detector) and intensity eg, [[x1,y1,I1], [x2, y2, I2], ...]
            The output is an 1d array with shape (100,) that record the response of each single pad detector
            '''
            network_input = (det_output-det_output.mean())/np.sqrt(det_output.var()) # normalization
            network_input = np.transpose(network_input) #!20220509
            network_input = network_input.reshape(1,-1)
            #print('network1')
            #print(network_input)
            #print(type(network_input))
            network_input = torch.from_numpy(network_input).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
            #print('network2')
            #print(network_input)
            #print(type(network_input))

            #predict=model(network_input).detach().cpu().numpy().reshape(-1)
            predict=model(network_input).detach().cpu().numpy().reshape(-1)

            #=========================#!20220509
            xdata_original=det_output.reshape(4, 4)  #!size-change
            #xdata_original=np.transpose(xdata_original)
            ydata=get_output([x, y])
            pred_out = 9*(np.argmax(predict)-20)    #!20220509
            predout_record.append([step, pred_out])
            print("Result: " + str(pred_out) + " deg")
            #plt.imshow(xdata_original, interpolation='nearest')  #, cmap="plasma")
            #ds, ag = d, angle*180/np.pi
            #plt.title('R_dist: ' + str(round(ds, 5)) + ' [m],  R_angle: ' + str(round(ag, 5)) + '[deg]\nP_angle: ' + str(pred_out))
            #plt.xlabel('y')    #!20220502 
            #plt.ylabel('x')    #!20220502 
            #plt.colorbar()
            #plt.savefig(predictpath + '/' + 'STEP%.3d.pkl'%step + "_predict.png")
            #plt.close()
            
            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(40,20))    #!20220520
            #fig1 = plt.figure(figsize=(10,10))
            #ax1 = fig1.set_subplot(121, frameon=False)
            #xdata_show = np.flip(np.transpose(xdata_original), 0)
            xdata_show = np.flip(xdata_original, 0)
            xdata_show = np.flip(xdata_show, 1)
            ax1.imshow(xdata_show, interpolation='nearest', cmap="plasma")
            #ds, ag = filename[:-5].split('_')
            ds, ag = d, angle*180/np.pi
            #ax1.set_title('R_dist: ' + str(ds) + ',  R_angle: ' + str(round(ag, 5)) + '\nP_angle: ' + str(pred_out))   #Mean_max: ' + str(max) + '\nStdev_max: ' + str(stdev_max))
            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)
            plt.xlabel('x')    #!20220502 out
            plt.ylabel('y')    #!20220502 out
            #ax1.set_xlabel('y')    #!20220502 
            #ax1.set_ylabel('x')    #!20220502 
            ax1.set_xlabel('x')    #!20220520 
            ax1.set_ylabel('y')    #!20220520 
            #plt.colorbar(ax1, colorbar()
            theta = np.linspace(-90, 270, 40)
            output1 = predict
            output2 = ydata.tolist()
            #fig2 = plt.figure(figsize=(10,10))
            #ax2 = fig2.add_subplot(122, frameon=False)
            for i in range(len(theta)-1):
                ax2.add_artist(
                    Wedge((0, 0), 1, theta[i], theta[i+1], width=0.3, color=(1, 1-output1[i], 1-output1[i])),
                )
                ax2.add_artist(
                    Wedge((0, 0), 0.7, theta[i], theta[i+1], width=0.3, color=(1-output2[i], 1-output2[i], 1)),
                )

            c1 = plt.Circle((0, 0), 1, color='k', fill=False)
            c2 = plt.Circle((0, 0), 0.7, color='k', fill=False)
            c3 = plt.Circle((0, 0), 0.4, color='k', fill=False)
            ax2.add_patch(c1)
            ax2.add_patch(c2)
            ax2.add_patch(c3)
            #ax2.text(5, 5, 'Real: ' + str(round(float(ag)), 5) + '\nPredict: ' + str(pred_out))
            #ax2.wedgeprops={"edgecolor":"0",'linewidth': 1,
                        #'linestyle': 'solid', 'antialiased': True}
            ax2.set_xlim((-1.2,1.2))
            ax2.set_ylim((-1.2,1.2))
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            #ax2.box(False)
            #ax.set_linestyle("-")
            #fig.show()
            #fig.savefig("savefig/angle_v2.11.png")
        
        
            #fig.savefig(predictpath + '/' + filename[:-4] + "png")
            #fig.suptitle('STEP%.3d'%step +'\nReal Angle: ' + str(round(ag, 5)) + ', \nPredicted Angle: ' + str(pred_out) + ' [deg]', fontsize=16) 
            fig.suptitle('Real Angle: ' + str(round(ag, 5)) + ', \nPredicted Angle: ' + str(pred_out) + ' [deg]', fontsize=60) 
            fig.savefig(predictpath + '/' + 'STEP%.3d'%step + "_predict.png")
            plt.close(fig)
            #=========================

            print(step,'simulation end')

        if record_data:#predict #!20220331
            data_dump={
            'RSID':RSID,
            'hxTrue':hxTrue,
            'det_output':det_output,
            'predict_list':predict
            }
            with open(os.path.join(recordpath,'STEP%.3d.pkl'%step),'wb') as f:
                pkl.dump(data_dump,f)

            try:    #!20220510
                with open(os.path.join(recordpath,'STEP%.3d.pkl'%step),'rb') as f:
                    data=pkl.load(f, encoding="latin1")
                print('File pickle success: ' + 'STEP%.3d.pkl'%step)
            except EOFError:
                print('EOFError: ' + 'STEP%.3d.pkl'%step)
                     

    rel_source = np.array(relsrc) #!20220509
    plt.plot(rel_source[:,0], rel_source[:,1])
    plt.title('relative source position')
    plt.savefig('mapping_data/save_fig/'+ file_header + '_rel_source.png')
    plt.close()
    
    xTrue_data = np.array(xTrue_record) #!20220509
    plt.plot(xTrue_data[:,0], xTrue_data[:,1])
    plt.title('xTrue: robot trajectory')
    plt.savefig('mapping_data/save_fig/'+ file_header + '_xTrue.png')
    plt.close()
    
    d_data = np.array(d_record) #!20220509
    plt.plot(d_data[:,0], d_data[:,1])
    plt.title('distance')
    plt.xlabel('step')
    plt.ylabel('distance [m]')
    plt.savefig('mapping_data/save_fig/'+ file_header + '_dist.png')
    plt.close()

    angle_data = np.array(angle_record) #!20220509
    predout_data = np.array(predout_record) #!20220509
    plt.plot(angle_data[:,0], angle_data[:,1], label="angle")
    plt.plot(predout_data[:,0], predout_data[:,1], label="pred_out")
    plt.title('angle')
    plt.xlabel('step')
    plt.ylabel('angle [deg]')
    plt.legend(loc="upper right")
    plt.savefig('mapping_data/save_fig/'+ file_header + '_angle.png')
    plt.close()

    #hxTrue_data = hxTrue_record[-1][:, 1:] #!20220509
    #plt.plot(hxTrue_data[0,:], hxTrue_data[1, :])
    #plt.savefig('save_fig/'+ file_header + '_hxTrue.png')
    #plt.close()

    with imageio.get_writer('mapping_data/save_fig/'+file_header + '.gif', mode='I') as writer:
        for figurename in sorted(os.listdir(predictpath)):
            image = imageio.imread(predictpath + "/" +figurename)
            writer.append_data(image)


    pass

#%%
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
    materials.cross_sections = '/home/rokabe/data1/openmc/endfb71_hdf5/cross_sections.xml'
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
    min_x3 = openmc.XPlane(x0=-2, boundary_type='transmission')     #!20220124 #!size-change
    max_x3 = openmc.XPlane(x0=+2, boundary_type='transmission')
    min_y3 = openmc.YPlane(y0=-2, boundary_type='transmission')
    max_y3 = openmc.YPlane(y0=+2, boundary_type='transmission')
    
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
    assembly.lower_left = [-1 * 4 / 2.0] * 2    #!20220124 #!size-change
    assembly.universes = [[cell_universe] * 4] * 4 #!size-change

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
    mesh.dimension = [4, 4]  #!size-change
    mesh.lower_left = [-2, -2]  #[-10, -10]     #!20220124  #!size-change
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


def process_aft_openmc_v1(folder1='random_savearray/', file1='detector_1source_20220118.txt', \
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
    mean = fiss['mean'].values.reshape((4, 4)) # numpy array   #!20220118    #!size-change
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
    stdev = absorb.std_dev.reshape((4, 4))    #!size-change
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
    mean_list=mean.T.reshape((1, 16)).tolist()  #!size-change
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
    #ds, ag = file2[:-5].split('_') #!20220509 out
    ds, ag = file2[8:-5].split('_') #!20220509
    plt.title('dist: ' + ds + ',  angle: ' + ag + '\nMean_max: ' + str(max) + '\nStdev_max: ' + str(stdev_max))
    plt.xlabel('y')  #!20220502 Adjust the incorrect axis setting!
    plt.ylabel('x')  #!20220502 Adjust the incorrect axis setting!
    plt.colorbar()
    #plt.show()   #!20220117
    #plt.savefig('random_savefig/abs_rate_20220118_6.png')   #!20220117
    plt.savefig(folder2 + file2) #   'random_savefig/abs_rate_20220118_6.png')   #!20220117
    plt.close()
    
    print('json dir')
    print(folder1+file1)
    
    print('fig dir')
    print(folder2+file2)
    return mean #!20220331


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

def after_openmc(rad_dist, rad_angle, folder1, folder2, header):      
        #folder1='random_savearray/'
        #file1=str(round(rad_dist[0], 5)) + '_' + str(round(rad_angle[0], 5)) + '.json'
        #file1=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '_' + str(idx) + '_' + str(j)+ '.json'
        #file1=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '_' + str(idx) + '_' + str(num_particles)+ '.json'
        #folder2='random_savefig/'
        #file2=str(round(rad_dist[0], 5)) + '_' + str(round(rad_angle[0], 5)) + '.png'
        #file2=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '_' + str(idx) + '_' + str(j)+ '.png'
        #file2=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '_' + str(idx) + '_' + str(num_particles)+ '.png'
    #folder1='openmc/discrete_data_20220507_v1.1/'
    #file1=str(round(rad_dist[0], 5)) + '_' + str(round(rad_angle[0], 5)) + '.json'
    #file1=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '.json'
    #file2=str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '.png'
    file1=header + "_" + str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '.json'
    file2=header + "_" + str(round(rad_dist, 5)) + '_' + str(round(rad_angle, 5)) + '.png'

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
    
    theta=rad_angle*np.pi/180
    rad_x, rad_y=[float(rad_dist*np.cos(theta)), float(rad_dist*np.sin(theta))]   #!20220119
            
    mm = process_aft_openmc_v1(folder1, file1, folder2, file2, rad_x, rad_y, norm=True)  #!20220201 #!20220119
    
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
    
    return mm #!20220508


#%%
if __name__ == '__main__':
    main()


