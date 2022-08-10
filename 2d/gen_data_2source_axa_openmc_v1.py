#%%
import numpy as np
import math
import matplotlib.pyplot as plt
import sys,os
import pickle as pkl

#sys.path.append('../')
sys.path.append('./')   #!20220331
from train_torch_openmc_tetris_v1 import *  #!20220717
#from gen_openmc_uniform_v2 import *
#from gen_openmc_data_discrete_v1 import *

from cal_param_axa_v1 import *

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

#shape_name = '2x2'
file_header = f"A20220809_5x5_v1.1"
recordpath = f'mapping_data/mapping_{file_header}'
model_path = f'save_model/model_openmc_5x5_ep2000_bs256_20220809_v1.1_model.pt'
model =torch.load(model_path)
seg_angles = 128
a_num = 5

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

DT = 0.1  # time tick [s]
SIM_TIME = 70.0
STATE_SIZE = 3
LM_SIZE = 3
#RSID = np.array([[0.0,5.0,5000000],[10.0,5.0,10000000]])   # source location (in detector frame) and intensity
#RSID = np.array([0.0,5.0,100000000])   #1 source
#RSID = np.array([10.0,20.0,100000000])   #1 source
#RSID = np.array([0.0,10.0,100000000])   #1 source
#RSID = np.array([5.0,5.0,100000000])   #1 source
#RSID = np.array([[0.0,1.0,0.5e6],[0.0,10.0,0.5e6]])    # 2 sources, #!20220804
RSID = np.array([[1.0,2.0,0.5e6],[-3.0,14.0,0.5e6]])  
source_energies = [0.5e6, 0.5e6]

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
#def openmc_simulation_uniform(source, header):
def openmc_simulation_uniform(sources_d_th, header, seg_angles):  #!20220717

    num_particles = 50000 #50000

    num_sources = len(sources_d_th)
    for i in range(num_sources):
        sources_d_th[i][0] *= 100
    # rad_x= source[0]*100    #!20220509 use [cm] instead of m in openmc
    # rad_y= source[1]*100
    # rad_dist = np.sqrt(rad_x**2 + rad_y**2) #!20220331 I need to change it later..
    # rad_angle = (np.arccos(rad_x/rad_dist)*180/np.pi)%360
    #use_panels = get_tetris_shape(shape_name)   #!20220717
    #gen_materials_geometry_tallies(panel_density, e_filter_tf, energy_filter_range)
    #get_output([rad_x, rad_y])
    #gen_settings(src_energy=src_E, src_strength=src_Str,  en_a=energy_a, en_b=energy_b, num_particles=num_particles, batch_size=batches, source_x=rad_x, source_y=rad_y)
    #before_openmc(rad_dist, rad_angle, num_particles)   #!20220508
    #before_openmc(rad_dist, rad_angle, num_particles, seg_angles)
    before_openmc(a_num, sources_d_th, num_particles, seg_angles)  #!20220803
    openmc.run()
    #mm = after_openmc(rad_dist, rad_angle, jsonpath, figurepath, seg_angles, header)
    mm = after_openmc(a_num, sources_d_th, jsonpath, figurepath, seg_angles, header) 
    return mm

#%%
def main(seg_angles):
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
            source_list=[] #!20220331
            relsrc_line = []    #!20220516
            d_record_line = [step]    #!20220516
            angle_line = [step]    #!20220516
            dist_ang_list = []
            sources_d_th = []   #!20220804
            sources_x_y_c = []
            #source=[]
            for i in range(RSID.shape[0]):
                # convert source location in world frame to detector frame
                dx = RSID[i, 0] - xTrue[0, 0]
                dy = RSID[i, 1] - xTrue[1, 0]
                d = math.sqrt(dx**2 + dy**2) # * 100 # unit m > ch    #!20220804
                angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
                print('RSID:', [RSID[i, 0],RSID[i, 1]])
                print('xTrue:', [xTrue[0, 0],xTrue[1, 0]])
                print('d:', d)
                print('[dx, dy]:', [dx, dy])
                print('angle (rad): ', angle)
                print('angle (deg): ', angle*180/np.pi)
                print('xTrue_ang: ', xTrue[2, 0])

                x=d*np.cos(angle)
                y=d*np.sin(angle)

                rate=RSID[i,2]

                source_list.append([x,y,rate])
                
                src_xy = [x, y] #[100*x, 100*y]
                source_x_y_c = {}
                source_x_y_c['position']=src_xy
                source_x_y_c['counts']=source_energies[i]   #RSID[i,2]
                sources_x_y_c.append(source_x_y_c)
                
                relsrc_line.append(x)
                relsrc_line.append(y)
                
                d_record_line.append(d)
                angle_line.append(angle*180/np.pi)
                dist_ang_list.append([d, angle*180/np.pi])
                #sources_d_th.append([100*d, angle*180/np.pi, source_energies[i]])
                sources_d_th.append([d, angle*180/np.pi, source_energies[i]])   #!20220804

            relsrc.append(relsrc_line)  #!20220509
            d_record.append(d_record_line)
            angle_record.append(angle_line)
            dist_ang = np.transpose(np.array(dist_ang_list))


            print(step,'simulation start')
            print("source")
            print(source_list)

            det_output=openmc_simulation_uniform(sources_d_th, 'STEP%.3d'%step, seg_angles)   #!20220717
            print('det_output')
            print(type(det_output))
            print(det_output.shape)
            '''
            The simulation function simulate the detector response given the radiation source position and intensity.
            The input of this function is a list of sources location (in detector frame, (0,0) is the center of the detector) and intensity eg, [[x1,y1,I1], [x2, y2, I2], ...]
            The output is an 1d array with shape (100,) that record the response of each single pad detector
            '''
            network_input = (det_output-det_output.mean())/np.sqrt(det_output.var()) # normalization
            network_input = np.transpose(network_input) #!20220509
            network_input = network_input.reshape(1,-1)

            network_input = torch.from_numpy(network_input).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

            predict=model(network_input).detach().cpu().numpy().reshape(-1)

            #=========================#!20220509
            xdata_original=det_output.reshape(a_num, a_num) #reshape(2, 3) #reshape(3, 2)  #reshape(2, 2)  #!size-change (x, y) #!20220804
            #xdata_original=np.transpose(xdata_original)
            #ydata=get_output([x, y], seg_angles)    #!20220729
            ydata = get_output_2source(sources_x_y_c, seg_angles)
            pred_out = (360/seg_angles)*(np.argmax(predict)-seg_angles/2)    #!20220729
            predout_record.append([step, pred_out])
            print("Result: " + str(pred_out) + " deg")

            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(40,20))    #!20220520

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
            #theta = np.linspace(-90, 270, 40)
            theta = np.linspace(-90, 270, seg_angles)   #!20220729
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

            c1 = plt.Circle((0, 0), 1, color='k', lw=5, fill=False)   #!20220729
            c2 = plt.Circle((0, 0), 0.7, color='k', lw=5, fill=False)
            c3 = plt.Circle((0, 0), 0.4, color='k', lw=5, fill=False)
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
            fig.savefig(predictpath + '/' + 'STEP%.3d'%step + "_predict.pdf")   #!20220729
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
    plt.savefig('mapping_data/save_fig/'+ file_header + '_rel_source.pdf')
    plt.close()
    
    xTrue_data = np.array(xTrue_record) #!20220509
    plt.plot(xTrue_data[:,0], xTrue_data[:,1])
    plt.title('xTrue: robot trajectory')
    plt.savefig('mapping_data/save_fig/'+ file_header + '_xTrue.png')
    plt.savefig('mapping_data/save_fig/'+ file_header + '_xTrue.pdf')
    plt.close()
    
    d_data = np.array(d_record) #!20220509
    plt.plot(d_data[:,0], d_data[:,1])
    plt.title('distance')
    plt.xlabel('step')
    plt.ylabel('distance [m]')
    plt.savefig('mapping_data/save_fig/'+ file_header + '_dist.png')
    plt.savefig('mapping_data/save_fig/'+ file_header + '_dist.pdf')
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
    plt.savefig('mapping_data/save_fig/'+ file_header + '_angle.pdf')
    plt.close()

    #hxTrue_data = hxTrue_record[-1][:, 1:] #!20220509
    #plt.plot(hxTrue_data[0,:], hxTrue_data[1, :])
    #plt.savefig('save_fig/'+ file_header + '_hxTrue.png')
    #plt.close()

    with imageio.get_writer('mapping_data/save_fig/'+file_header + '.gif', mode='I') as writer:
        for figurename in sorted(os.listdir(predictpath)):
            if figurename.endswith('png'):  #!20220729
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

def gen_materials_geometry_tallies(a_num, panel_density):    #!20220803
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
    min_x3 = openmc.XPlane(x0=-a_num/2, boundary_type='transmission')     #!20220124 size-change!
    max_x3 = openmc.XPlane(x0=+a_num/2, boundary_type='transmission')
    min_y3 = openmc.YPlane(y0=-a_num/2, boundary_type='transmission')
    max_y3 = openmc.YPlane(y0=+a_num/2, boundary_type='transmission')
    
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
    assembly.lower_left = [-1 * a_num / 2.0] * 2    #!20220626  # size-change!
    assembly.universes = [[cell_universe] * a_num] * a_num  #! size-change!

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
    plt.savefig('save_fig/geometry_20220201.png')   #!20220117
    plt.savefig('save_fig/geometry_20220201.pdf')
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
    mesh.dimension = [a_num, a_num]   #! size-change!
    mesh.lower_left = [-a_num/2, -a_num/2]  #[-10, -10]     #!20220124   #! size-change!
    mesh.width = [1, 1] #[2, 2]  #!20220124

    # Instantiate tally Filter
    mesh_filter = openmc.MeshFilter(mesh)

    # Instantiate energy Filter
    #energy_filter = openmc.EnergyFilter([0, 0.625, 20.0e6])
    # Instantiate the Tally
    tally = openmc.Tally(name='mesh tally')
    
    # if e_filter:
    #     energy_filter = openmc.EnergyFilter(*energy)    #!20220204
    #     tally.filters = [mesh_filter, energy_filter]    #!20220204
    
    #else:
    if True:
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
#def gen_settings(src_energy=None, src_strength=1, en_source=1e6, en_prob=1, num_particles=10000, batch_size=100, source_x=rad_source_x[0], source_y=rad_source_x[1]): #!20220224
def gen_settings(src_energy, src_strength, en_prob, num_particles, batch_size, sources): #!20220803
    # Create a point source
    #point = openmc.stats.Point((2, 13, 0))
    #source = openmc.Source(space=point)
    #point1 = openmc.stats.Point((30, 13, 0))
    #point1 = openmc.stats.Point((rad_source1[0], rad_source1[1], 0))
    num_sources = len(sources)
    sources_list = []
    for i in range(num_sources):
        point = openmc.stats.Point((sources[i]['position'][0], sources[i]['position'][1], 0))
        #source1 = openmc.Source(space=point1, particle='photon')  #!20220118
        source = openmc.Source(space=point, particle='photon', energy=src_energy, strength=src_strength)  #!20220204    #!20220118
        source.energy = openmc.stats.Discrete(x=(sources[i]['counts']), p=en_prob)
        sources_list.append(source) #, source2, source3]     #!20220118
    #point2 = openmc.stats.Point((-50, 6, 0))
    #source2 = openmc.Source(space=point2, particle='photon')  #!20220118
    #point3 = openmc.stats.Point((1, -20, 0))
    #source3 = openmc.Source(space=point3, particle='photon')  #!20220118
    #source.particle = 'photon'  #!20220117

    #!==================== 20220223
    #source1.energy = openmc.stats.Uniform(a=en_a, b=en_b)
    #source1.energy = openmc.stats.Discrete(x=en_source, p=en_prob)
    #!====================

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'  #!20220118
    settings.photon_transport = True  #!20220117
    #settings.electron_treatment = 'led' #!20220117
    #settings.source = source
    settings.source = sources_list #, source2, source3]     #!20220118
    settings.batches = batch_size #100
    settings.inactive = 10
    settings.particles = num_particles

    settings.export_to_xml()

    #os.system("cat settings.xml")


def run_openmc():
    # Run OpenMC!
    openmc.run()


def process_aft_openmc(a_num, folder1, file1, folder2, file2, sources, seg_angles, norm):
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
    mean = fiss['mean'].values.reshape((a_num, a_num)) # numpy array   #!20220118   #! size-change!
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
    stdev = absorb.std_dev.reshape((a_num, a_num))     #! size-change!
    stdev_max = stdev.max()
    #print(stdev)

    #==================================
    num_sources = len(sources)
    
    data_json={} #!20220119
    data_json['source']=sources #!20220803
    #print('source: ' + str(type([source_x, source_y])))
    data_json['intensity']=100   #!20220119 tentative value!
    data_json['miu_detector']=0.3   #!20220119 constant!
    data_json['miu_medium']=1.2   #!20220119 constant!
    data_json['miu_air']=0.00018   #!20220119 constant!
    data_json['output']=get_output_2source(sources, seg_angles).tolist()#get_output(sources, seg_angles).tolist()    #!20220803
    data_json['num_sources']=num_sources    #!20220803
    data_json['seg_angles']=seg_angles
    #print('output: ' + str(type(data_json['output'])))
    data_json['miu_de']=0.5   #!20220119 constant!
    mean_list=mean.T.reshape((1, a_num**2)).tolist()    #! size-change!
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
    #ds, ag = file2[:-5].split('_')[1:]  #!20220517
    ds_ag_list = file2[:-5].split('_')[1:]  #!20220517
    ds_ag_title = ''
    for i in range(num_sources):
        ds, ag = ds_ag_list[2*i], ds_ag_list[2*i+1]
        ds_ag_line = f'dist{i}: {ds},  angle{i}: {ag}'
        if i != num_sources-1:
            ds_ag_line += '\n'
        ds_ag_title += ds_ag_line
    plt.title(ds_ag_title)
    plt.xlabel('y')  #!20220502 Adjust the incorrect axis setting!
    plt.ylabel('x')  #!20220502 Adjust the incorrect axis setting!
    plt.colorbar()
    #plt.show()   #!20220117
    #plt.savefig('random_savefig/abs_rate_20220118_6.png')   #!20220117
    plt.savefig(folder2 + file2) #   'random_savefig/abs_rate_20220118_6.png')   #!20220117
    plt.savefig(folder2 + file2[:-3] + 'pdf')
    plt.close()
    print('json dir')
    print(folder1+file1)
    
    print('fig dir')
    print(folder2+file2)
    return mean #!20220717


#def get_output(source):
def get_output(source, num):    #!20220728
    #sec_center=np.linspace(-np.pi,np.pi,41)
    sec_center=np.linspace(-np.pi,np.pi,num+1)
    output=np.zeros(num)#(40)
    sec_dis=2*np.pi/num #40.
    angle=np.arctan2(source[1],source[0])
    before_indx=int((angle+np.pi)/sec_dis)
    if before_indx>=num: #!20220430 (actually no need to add these two lines..)
        before_indx-=num
    after_indx=before_indx+1
    if after_indx>=num:
        after_indx-=num
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

def get_output_2source(sources, num):
    sec_center=np.linspace(-np.pi,np.pi,num+1)
    output=np.zeros(num)
    sec_dis=2*np.pi/num
    ws=np.array([np.mean(source["counts"]) for source in sources])
    print('ws_point1')   #!20220303
    ws=ws/ws.sum()
    for i in range(len(sources)):
        source =sources[i]
        angle=np.arctan2(source["position"][1],source["position"][0])
        before_indx=int((angle+np.pi)/sec_dis)
        after_indx=before_indx+1
        if after_indx>=40:
            after_indx-=40
        w1=abs(angle-sec_center[before_indx])
        w2=abs(angle-sec_center[after_indx])
        if w2>sec_dis:
            w2=abs(angle-(sec_center[after_indx]+2*np.pi))
            #print w2
        output[before_indx]+=w2/(w1+w2)*ws[i]
        output[after_indx]+=w1/(w1+w2)*ws[i]
    #print angle,sec_center[before_indx],sec_center[after_indx]
    return output


#def before_openmc(rad_dist, rad_angle, num_particles):
def before_openmc(a_num, sources_d_th, num_particles, seg_angles):  #!20220803
#if __name__ == '__main__':
    ###=================Input parameter======================
    #num_data = 100
    batches = 100
    panel_density = 5.76 #g/cm3
    src_E = None    #[1,3]
    src_Str = 10
    num_sources = len(sources_d_th)
    #num_particles = 5000
    #dist_min = 100
    #dist_max = 1000
    #dist = 100
    #angle = 0
    #idx = 112
    #energy = [0, 0.625, 20.0e6]     #!20220128
    #energy_filter_range = [0.1e6, 2e6]     #!20220803
    #e_filter_tf=False  #!20220803
    #source_energy = (0.5e6)
    energy_prob = (1)
    #energy = [7.5, 19]  
    ###=======================================
    start = timeit.timeit()
    start_time = datetime.now()
    
    #gen_materials_geometry_tallies(a_num, panel_density, e_filter_tf, energy_filter_range)     #!20220205
    gen_materials_geometry_tallies(a_num, panel_density)     #!20220803
    j=batches
    #for j in range(10,batches, 10):
    #for i in range(num_data):
    #rad_dist=dist   #np.random.randint(dist_min, dist_max) + np.random.random(1)    #!20220128
    #rad_dist=np.random.randint(dist_min, dist_max) + np.random.random(1)
        #rad_angle=angle  #np.random.randint(0, 359) + np.random.random(1)    #!20220128
    #rad_angle=np.random.randint(0, 359) + np.random.random(1)
    #theta=rad_angle*np.pi/180
        #rad_source=[float(rad_dist*np.cos(theta)), float(rad_dist*np.sin(theta))]
    sources = [] #float(rad_dist*np.cos(theta))
    for i in range(num_sources):
        theta=sources_d_th[i][1]*np.pi/180
        dist = sources_d_th[i][0]
        source = {}
        src_xy = [float(dist*np.cos(theta)), float(dist*np.sin(theta))]
        source['position']=src_xy
        source['counts']=sources_d_th[i][2]
        sources.append(source)
    #rad_x, rad_y=[float(rad_dist*np.cos(theta)), float(rad_dist*np.sin(theta))]   #!20220119
    #print([rad_x, rad_y])
    #get_output([rad_x, rad_y])
    #get_output([rad_x, rad_y], seg_angles)
    get_output_2source(sources, seg_angles) #!20220803
    
        #gen_settings(rad_sources1=rad_source)
    #gen_settings(src_energy=src_E, src_strength=src_Str,  en_source=source_energy, en_prob=energy_prob, num_particles=num_particles, batch_size=j, source_x=rad_x, source_y=rad_y)    #!20220224
    gen_settings(src_energy=src_E, src_strength=src_Str, en_prob=energy_prob, num_particles=num_particles, batch_size=j, sources=sources) 
        #gen_tallies()
        
        #openmc.run()
#openmc.run(mpi_args=['mpiexec', '-n', '4'])
        #openmc.run(mpi_args=['mpiexec', '-n', '4', "-s", '11'])
        #openmc.run(threads=11)

#def after_openmc(rad_dist, rad_angle):      
def after_openmc(a_num, sources_d_th, folder1, folder2, seg_angles, header):    #!20220517
    num_sources = len(sources_d_th)
    d_a_seq = ""
    for i in range(num_sources):
        d_a_seq += '_' + str(round(sources_d_th[i][0], 5)) + '_' + str(round(sources_d_th[i][1], 5))
    # file1=header + "_" + str(round(dist, 5)) + '_' + str(round(angle, 5)) + '.json'
    # file2=header + "_" + str(round(dist, 5)) + '_' + str(round(angle, 5)) + '.png'
    file1=header + d_a_seq + '.json'    #!20220803
    file2=header + d_a_seq + '.png'

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
    
    # theta=angle*np.pi/180
    # rad_x, rad_y=[float(dist*np.cos(theta)), float(dist*np.sin(theta))]   #!20220119

    sources = [] #float(rad_dist*np.cos(theta))
    for i in range(num_sources):
        theta=sources_d_th[i][1]*np.pi/180
        dist = sources_d_th[i][0]
        source = {}
        src_xy = [float(dist*np.cos(theta)), float(dist*np.sin(theta))]
        source['position']=src_xy
        source['counts']=sources_d_th[i][2]
        sources.append(source)

    #mm = process_aft_openmc(folder1, file1, folder2, file2, rad_x, rad_y, seg_angles, norm=True)  #!20220201 #!20220119
    mm = process_aft_openmc(a_num, folder1, file1, folder2, file2, sources, seg_angles, norm=True)  #!20220803

    return mm #!20220508



#sys.path.append('../')
sys.path.append('./')   #!20220331
#from geo import *
from geo_v1 import *   #!20220206

def point_between_rays(p,r1,r2):
    '''
    r1->r2 counterclockwise, r1, r2 must have same source points
    '''
    p0=r1.p1

    theta=np.arctan2(p.y-p0.y,p.x-p0.x)

    if r2.arg>=r1.arg:
        if theta<=r2.arg and r1.arg<=theta:
            return True
        else:
            return False
    else:
        arg1=r1.arg
        arg2=r2.arg+2*np.pi

        if theta < r1.arg:
            return False
        if theta > r2.arg:
            return False
        return True
    pass
#%%
def clean_point_list(ps,err=1e-10):
    new_ps=[]
    for i in range(len(ps)):
        overlap=False
        # for j in range(0,i):
        #     #if i==j:continue
        #     if getLineLength(ps[i],ps[j])<=err:
        #         overlap=True
        for pp in new_ps:
            if getLineLength(ps[i],pp)<=err:
                overlap=True
                break
        if not overlap:
            new_ps.append(ps[i])
    return new_ps

#%%
def sort_poly_apex(ps):

    ps=clean_point_list(ps)
    
    p0=min(ps,key=lambda p: p.x)

    ps.remove(p0)

    ps=sorted(ps,key=lambda p:np.arctan2(p.y-p0.y,p.x-p0.x))

    ps=[p0]+ps

    return ps

#%%
def cal_tri_area(ps):
    #a= np.sqrt(ps[0].x-ps[1]**2)
    ps=ps+[ps[0]]
    els=[]
    for i in range(len(ps)-1):
        sb=np.sqrt((ps[i].x-ps[i+1].x)**2+(ps[i].y-ps[i+1].y)**2)
        els.append(sb)

    a,b,c=els
    if a<=1e-6 or b<=1e-6 or c<=1e-6:
        return 0.
    p = (a + b + c) / 2
    #print a,b,c,ps[0:3]
    S=(p*(p - a)*(p - b)*(p - c))** 0.5
    #print S,ps[0:3]
    return  S

#%%
def cal_area(ps):
    #ps=poly.apex

    if len(ps)>3:
        area=0.
        for i in range(1,len(ps)-1):
            area+=cal_tri_area([ps[0],ps[i],ps[i+1]])
            #print area,ps[0],ps[i],ps[i+1]
        return area
    elif len(ps)==3:
        return cal_tri_area(ps)
    else:
        return 0.

#%%
class Square(Polygon):
    """docstring for Square"""
    def __init__(self, center,a):
        if type(center) is Point(0,0):
            self.center=center
        else:
            self.center=Point(center[0],center[1])
        if (type(a) is type(1.0)) or (type(a) is type(1)):
            self.a=float(a)
            self.b=float(a)
        else:
            self.a=float(a[0])
            self.b=float(a[1])

        a=self.a
        b=self.b

        apex=[Point(center[0]+a/2.,center[1]+b/2.),
        Point(center[0]-a/2.,center[1]+b/2.),
        Point(center[0]-a/2.,center[1]-b/2.),
        Point(center[0]+a/2.,center[1]-b/2.)]
        super(Square, self).__init__(apex)
        #self.arg = arg

    def contain_point(self,pp,err=1e-10):
        apex=sort_poly_apex(self.apex)

        s1=cal_area([pp]+apex+[apex[0]])

        s2=cal_area(apex)

        if abs(s1-s2)<err:
            return True
        else:
            return False
        
#%%

def square_intersect(sq,r1,r2):
    points=[]
    for ap in sq.apex:
        if point_between_rays(ap,r1,r2):
            points.append(ap)
    #print 'pp',points
    points+=cross_line_polygon2(r1,sq)
    #print 'pp',points
    points+=cross_line_polygon2(r2,sq)

    #print 'cc', cross_line_polygon2(r2,sq),r2.arg
    #print sq.apex
    #print 'pp',points
    #print points
    if not points:
        return 0.
    if sq.contain_point(r1.p1):
        points.append(r1.p1)
    points=sort_poly_apex(points)

    #print 'pp',points

    a1=cal_area(points)

    a2=cal_area(sq.apex)

    return a1/float(a2)

    pass
#%%
class Map(object):
    """docstring for Map"""
    def __init__(self, x_info, y_info):
        super(Map, self).__init__()
        
        self.x_num=x_info[2]
        self.y_num=y_info[2]

        self.size=self.x_num*self.y_num

        x_min=x_info[0]
        x_max=x_info[1]

        y_min=y_info[0]
        y_max=y_info[1]

        self.dx=(x_info[1]-x_info[0])/float(self.x_num)
        self.dy=(y_info[1]-y_info[0])/float(self.y_num)

        self.center_list=[]
        self.square_list=[]

        for i in range(self.x_num):
            for j in range(self.y_num):  
                 x= x_min + self.dx/2 + i * self.dx
                 y= y_min + self.dy/2 + j * self.dy

                 self.center_list.append([x,y])
                 self.square_list.append(Square([x,y],[self.dx,self.dy]))

        self.intensity_list=np.zeros(self.x_num*self.y_num)
        self.intensity=self.intensity_list.reshape((self.x_num,self.y_num))

#%%
def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def cal_cji(m,pose,out_size=40):
    '''
    m: map
    pose:[x,y,theta]
    '''
    #yj=np.zeros((out_size,1))
    cji=np.zeros((out_size,m.size))
    pose=pose.reshape((3,1))

    dtheta=2*np.pi/out_size

    for j in range(out_size):
        start_angle=pi_2_pi(pose[2,0]-np.pi+j*dtheta)
        end_angle=pi_2_pi(start_angle+dtheta)

        ss=Point(pose[0,0],pose[1,0])
        r_start=Ray(ss,start_angle)
        r_end=Ray(ss,end_angle)

        for i in range(m.size):
            #print j,i
            cji[j,i]=square_intersect(m.square_list[i],r_start,r_end)
            
            dx=m.center_list[i][0]-pose[0,0]
            dy=m.center_list[i][1]-pose[1,0]
            r=np.sqrt(dx**2+dy**2)

            # observation matrix
            cji[j,i]=cji[j,i]/r     # 2D
            # cji[j,i]=cji[j,i]/r**2  # 3D


    return cji

def cal_yj(input_data,output_data):
    yj=output_data*np.mean(input_data)
    return yj

    pass
#%%
def test(seg_angles):   #!20220729
    plt.figure()

    cmap = matplotlib.cm.get_cmap('gray')

    m=Map([-5,5,10],[-5,5,10])
    pose=np.array([2.2,2.2,np.pi/2]).reshape((3,1))

    cji=cal_cji(m,pose,seg_angles)

    j=5

    dtheta=2*np.pi/40.
    start_angle=pi_2_pi(pose[2,0]-np.pi+j*dtheta)
    end_angle=pi_2_pi(start_angle+dtheta)

    ss=Point(pose[0,0],pose[1,0])
    r_start=Ray(ss,start_angle)
    r_end=Ray(ss,end_angle)
    print(cji[j,:]) #!20220206
    for i in range(m.size):#m.size

        m.square_list[i].plot(ax=plt.gca(), facecolor=cmap(1-cji[j,i]))
        #print cji[j,i]

    r_start.plot(8,color='r')
    r_end.plot(8,color='b')
    #print m.center_list
        #self.arg = arg
        
def test2():
    plt.figure()

    cmap = matplotlib.cm.get_cmap('gray')

    m=Map([-5,5,10],[-5,5,10])
    pose=np.array([2.,2,np.pi/2]).reshape((3,1))

    j=35
    i=66



    dtheta=2*np.pi/40.
    start_angle=pi_2_pi(pose[2,0]-np.pi+j*dtheta)
    end_angle=pi_2_pi(start_angle+dtheta)

    ss=Point(pose[0,0],pose[1,0])
    print(m.square_list[i].contain_point(ss))    #!20220206
    r_start=Ray(ss,start_angle)
    r_end=Ray(ss,end_angle)
    c=square_intersect(m.square_list[i],r_start,r_end)

    m.square_list[i].plot()
    r_start.plot(10,color='r')
    r_end.plot(10,color='b')

    print('c',c)     #!20220206

    pass

#%%

def write_data(seg_angles, recordpath): #!20220729
    #recordpath = '../../../data/drd/mapping_0803' # NN prediction results are stored under this folder, each .pkl file stores one measurement (NN output and measurement poses)
    #recordpath = '../../../data/drd/mapping_20220322'   #!20220322
    #recordpath = 'mapping_20220322'   #!20220322
    #recordpath = 'mapping_0803_x'   #!20220322
    #recordpath = 'mapping_data/mapping_A20220627_v2.2'   #!20220331
    #recordpath = 'mapping_data/mapping_A20220510_v1.2'   #!20220331
    files=os.listdir(recordpath)
    files=sorted(files)
    #files.remove('STEP000.pkl') #!20220510

    #m=Map([-10,15,25],[-5,25,30])  #!20220516 out
    m=Map([-15,15,30],[-5,25,30])   #!20220516

    for filename in files:
        
        try:    #!20220510
            with open(os.path.join(recordpath,filename),'rb') as f:
                data=pkl.load(f, encoding="latin1")
            #print('File pickle success: ' + recordpath + '/'+filename)
        except EOFError:
            print('EOFError: ' + recordpath + '/'+ filename)
        #with open(os.path.join(recordpath,filename),'rb') as f:
            #data=pkl.load(f)
            #data=pkl.load(f, encoding="latin1")  #!20220316

        if data['det_output'] is None:
            #print("could not find det_output in " + filename)   #!20220509
            continue

        det_output=np.array(data['det_output'])
        predict=np.array(data['predict_list'])
        #plt.plot(np.linspace(-180,180,41)[0:40],np.array(predict).reshape(40))
        # plt.show()
        # raw_input()
        pose=data['hxTrue'][:,-1] # pose of the detector
        print(filename)  #!20220206

        cji=cal_cji(m,pose, seg_angles) #!20220729
        yj=cal_yj(det_output,predict)

        data['cji']=cji
        data['yj']=yj

        #with open(os.path.join(recordpath,filename),'wb') as f:
        if not os.path.isdir(recordpath+'_cal'):    #!20220510
            os.mkdir(recordpath+'_cal')
        #os.system('rm ' + recordpath+'_cal' + "/*")    #!20220509
        with open(os.path.join(recordpath+'_cal',filename),'wb') as f:
            pkl.dump(data,f)
            
        try:    #!20220510
            with open(os.path.join(recordpath+'_cal',filename),'wb') as f:
                pkl.dump(data,f)
            #print('File pickle success: ' + recordpath+'_cal/'+filename)
        except EOFError:
            print('EOFError: ' + recordpath+'_cal/'+filename)
        #print cji,yj
        #raw_input()
        #raw_input()

    pass




#%%
if __name__ == '__main__':
    main(seg_angles)
    write_data(seg_angles, recordpath)

