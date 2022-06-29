#%%
import numpy as np
import math
import matplotlib.pyplot as plt
import sys, os
import pickle as pkl


#sys.path.append('../')
sys.path.append('./')   #!20220331
#from train_torch_openmc import *    #!20220502 out!
from train_torch_openmc_debug_20220508 import *    #!20220502 out!
from gen_openmc_uniform_v2 import *

from cal_param_v1 import *
from dataset_20220508 import *
import torch
import copy
from matplotlib.patches import Wedge

#record_data=False
record_data=True

#============================= #!20220331
GPU_INDEX = 1#0
USE_CPU = False
# print torch.cuda.is_available()
if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda:%d"%GPU_INDEX) 
    torch.cuda.set_device(GPU_INDEX)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #!20220502

#DEFAULT_DEVICE = torch.device("cuda:2") 
#DEFAULT_DEVICE = torch.device("cpu") 
#DEFAULT_DTYPE = torch.double
#============================= #!20220331

#recordpath = 'mapping_0803' #?pkl files with python2 is stored
#recordpath = 'mapping_20220331'
#if __name__ == '__main__' and record_data:
#    if not os.path.isdir(recordpath):
#        os.mkdir(recordpath)

#jsonpath = recordpath + "_json"
#if __name__ == '__main__' and record_data:
#    if not os.path.isdir(jsonpath):
#        os.mkdir(jsonpath)

#figurepath = recordpath + "_figure"
#if __name__ == '__main__' and record_data:
#    if not os.path.isdir(figurepath):
#        os.mkdir(figurepath)

predictpath = 'predict_test/20220520_predict_v1.1'
#if __name__ == '__main__' and record_data:
#    if not os.path.isdir(predictpath):
#        os.mkdir(predictpath)

isExist1 = os.path.exists(predictpath)
if not isExist1:
# Create a new directory because it does not exist 
    os.makedirs(predictpath)
    print("The new directory "+ predictpath +" is created!")
#os.remove(predictpath + "*.png")


#-------


#--------




#model_path = '../2source_unet_model.pt'    #!20220331
#model_path = 'model_openmc_10cm_500000_ep4000_bs256_20220226_uniform_1.2_model.pt'
#model_path = 'model_openmc_10cm_500000_ep4000_bs256_20220410_uniform_3.1_model.pt'  #!20220502 out
model_path = 'save_model/model_20220508_openmc_10cm_50000_ep4000_bs256_uniform_new_test_v2.1_model.pt' #!20220502 out
model =torch.load(model_path)
print(model)    #!20220412

#data_path = 'openmc/data_20220226_1.2'  #!20220502 out
data_path = 'openmc/discrete_data_20220503_v1.1.1'  #!20220502

#DT = 0.1  # time tick [s]
#SIM_TIME = 50.0
##STATE_SIZE = 3
#LM_SIZE = 3
#RSID = np.array([[0.0,5.0,5000000],[10.0,5.0,10000000]])   # source location (in detector frame) and intensity
#RSID = np.array([0.0,5.0,100000000])   #1 source

#SIM_STEP=25

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

#def calc_input(time):
    '''
    a simplified open loop controller to make the robot move in a circle trajectory
    Input: time
    Output: control input of the robot
    '''

    if time <= 0.:  # wait at first
        v = 0.0
        yawrate = 0.0
    else:
        v = 1.0  # [m/s]
        yawrate = 0.1  # [rad/s]

    u = np.array([v, yawrate]).reshape(2, 1)

    return u

#def motion_model(x, u):
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

#def openmc_simulation_uniform(source):
    batches = 100
    panel_density = 5.76 #g/cm3
    src_E = None
    src_Str = 10
    num_particles = 5000
    energy_filter_range = [0.1e6, 2e6]
    e_filter_tf=False
    energy_a = 0.5e6
    energy_b = 1e6
    rad_x= source[0]
    rad_y= source[1]
    rad_dist = np.sqrt(rad_x**2 + rad_y**2) #!20220331 I need to change it later..
    rad_angle = (np.arccos(rad_x/rad_dist)*180/np.pi)%360
    
    gen_materials_geometry_tallies(panel_density, e_filter_tf, energy_filter_range)
    get_output([rad_x, rad_y])
    gen_settings(src_energy=src_E, src_strength=src_Str,  en_a=energy_a, en_b=energy_b, num_particles=num_particles, batch_size=batches, source_x=rad_x, source_y=rad_y)
    openmc.run()
    file1=str(round(rad_dist, 4)) + '_' + str(rad_angle) + '.json'
    file2=str(round(rad_dist, 4)) + '_' + str(rad_angle) + '.png'
    mm = process_aft_openmc(jsonpath+"/", file1, figurepath+"/", file2, rad_x, rad_y, norm=True) #norm=True)
    return mm


#!20220520 ===

#import matplotlib.pyplot as plt
#from matplotlib.patches import Wedge

#import numpy as np

#theta = np.linspace(0, 360, 40)
output1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6581555709371346, 0.34184442906286533, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
output2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.826303819917137, 0.17369618008286303, 0.0, 0.0, 0.0, 0.0, 0.0]

def plot_angle(theta, output1, output2):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, frameon=False)
    for i in range(len(theta)-1):
        ax.add_artist(
            Wedge((0, 0), 1, theta[i], theta[i+1], width=0.3, color=(1, 1-output1[i], 1-output1[i])),
        )
        ax.add_artist(
            Wedge((0, 0), 0.7, theta[i], theta[i+1], width=0.3, color=(1-output2[i], 1-output2[i], 1)),
        )

    c1 = plt.Circle((0, 0), 1, color='k', fill=False)
    c2 = plt.Circle((0, 0), 0.7, color='k', fill=False)
    c3 = plt.Circle((0, 0), 0.4, color='k', fill=False)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)
    ax.wedgeprops={"edgecolor":"0",'linewidth': 1,
                    'linestyle': 'solid', 'antialiased': True}
    ax.set_xlim((-2,2))
    ax.set_ylim((-2,2))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    #ax.set_linestyle("-")
    fig.show()
    fig.savefig("savefig/angle_v2.11.png")

#!====

def main():
    #time=0
    #step=-1
    

    
    #xTrue = np.zeros((STATE_SIZE, 1))
    #hxTrue = xTrue # pose (position and orientation) of the robot (detector) in world frame

    #while SIM_TIME >= time:
        #time += DT
        #step+=1

        
        #u = calc_input(time)

        #xTrue = motion_model(xTrue, u)


        #hxTrue = np.hstack((hxTrue, xTrue))

        #det_output=None
        #predict=None

        

        #if step%SIM_STEP==0:
            #print('STEP %d'%step)
            #source_list=[] #!20220331
            #source=[]
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
            #dx = RSID[0] - xTrue[0, 0]
            #dy = RSID[1] - xTrue[1, 0]
            #d = math.sqrt(dx**2 + dy**2)
            #angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
            #x=d*np.cos(angle)
            #y=d*np.sin(angle)
            #rate=RSID[2]
            #source.append(x,y,rate)
            #source=[x,y,rate]

            #print(step,'simulation start')
            #print("source")
            #print(source)
            #print(len(source))
            
            # replace this line with openmc simulation function
            #det_output=simulation(source_list)     #!20220331
            #det_output=openmc_simulation_uniform(source)
            #print('det_output')
            #print(det_output)
    '''
            The simulation function simulate the detector response given the radiation source position and intensity.
            The input of this function is a list of sources location (in detector frame, (0,0) is the center of the detector) and intensity eg, [[x1,y1,I1], [x2, y2, I2], ...]
            The output is an 1d array with shape (100,) that record the response of each single pad detector
    '''
            #network_input = (det_output-det_output.mean())/np.sqrt(det_output.var()) # normalization
            #network_input = network_input.reshape(1,-1)
            #print('network1')
            #print(network_input)
            #print(type(network_input))
            #network_input = torch.from_numpy(network_input).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
            #print('network2')
            #print(network_input)
            #print(type(network_input))
    #!20220410
    files= os.listdir(data_path)
    for filename in files:
        if not filename.endswith('.json'):continue
        with open(os.path.join(data_path,filename),'r') as f:
            data=json.load(f)
            names=filename
            xdata=data['input']
            xdata=np.array(xdata)
            #xdata=xdata.reshape(10, 10)
            xdata_original = copy.deepcopy(xdata)
            xdata=torch.from_numpy(xdata)
            #xdata_original = copy.deepcopy(xdata)
            xdata=xdata.cuda()
            ydata=get_output(data['source'])
            ydata=np.array(ydata)

        #?+++++++++++++++++
                #predict=model(network_input).detach().cpu().numpy().reshape(-1)
                #predict=model(network_input)#.detach().cpu().numpy().reshape(-1)
        #xdata.to(DEFAULT_DEVICE)
        #xdata.to(torch.device("cpu"))
        #xdata.to("cuda")
        print('```````````````````````')
        print("prediction starts here: " + filename)
        print(xdata)
        predict=model(xdata)
        print('"""""""""predict=model(xdata)"""""""""')
        print(predict)
        predict=predict.detach()
        print('"""""""""predict.detach()"""""""""')
        print(predict)
        predict=predict.cpu()
        print('"""""""""predict.cpu()"""""""""')
        print(predict)
        predict=predict.numpy()
        print('"""""""""predict.numpy()"""""""""')
        print(predict)
        predict=predict.reshape(-1)
        print('""""""""predict.reshape(-1)""""""""""')
        print(predict)
            #predict=model(network_input).detach().cpu().numpy()#.reshape(-1)
        #?+++++++++++++++++

            #print(step,'simulation end')
        print('+++++++++++++++++++++++++')
        print("prediction complete: " + filename)
        print(predict.shape)
        print(type(predict))
        print(predict)
        pred_out = 9*(np.argmax(predict)-20)
        print("Result: " + str(pred_out) + " deg")
        
        xdata_original=xdata_original.reshape(10, 10)
        xdata_original=np.transpose(xdata_original)
        #print(xdata_original)
        #print(xdata_original.shape)
        #plt.imshow(xdata_original, interpolation='nearest')  #, cmap="plasma")
        #ds, ag = filename[:-5].split('_')
        #plt.title('R_dist: ' + ds + ',  R_angle: ' + ag + '\nP_angle: ' + str(pred_out))   #Mean_max: ' + str(max) + '\nStdev_max: ' + str(stdev_max))
        #plt.xlabel('x')    #!20220502 out
        #plt.ylabel('y')    #!20220502 out
        #plt.xlabel('y')    #!20220502 
        #plt.ylabel('x')    #!20220502 
        #plt.colorbar()
        #plt.savefig(predictpath + '/' + filename[:-4] + "png")
        #plt.close()


        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,10))
        #fig1 = plt.figure(figsize=(10,10))
        #ax1 = fig1.set_subplot(121, frameon=False)
        xdata_show = np.flip(np.transpose(xdata_original), 0)
        ax1.imshow(xdata_show, interpolation='nearest', cmap="plasma")
        ds, ag = filename[:-5].split('_')
        ax1.set_title('R_dist: ' + ds + ',  R_angle: ' + ag + '\nP_angle: ' + str(pred_out))   #Mean_max: ' + str(max) + '\nStdev_max: ' + str(stdev_max))
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        plt.xlabel('x')    #!20220502 out
        plt.ylabel('y')    #!20220502 out
        ax1.set_xlabel('y')    #!20220502 
        ax1.set_ylabel('x')    #!20220502 
        #plt.colorbar(ax1, colorbar()
        theta = np.linspace(-180, 180, 40)
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
        
        
        fig.savefig(predictpath + '/' + filename[:-4] + "png")
        plt.close(fig)



        #if record_data:#predict #!20220331
            #data_dump={
            #'RSID':RSID,
            #'hxTrue':hxTrue,
            #'det_output':det_output,
            #'predict_list':predict
            #}
            #with open(os.path.join(recordpath,'STEP%.3d.pkl'%step),'wb') as f:
                #pkl.dump(data_dump,f)



    pass

#%%
if __name__ == '__main__':
    main()



# %%
