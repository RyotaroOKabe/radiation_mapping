#%%
# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import sys,os
import pickle as pkl
sys.path.append('./')   #!20220331
from utils.model import * 
import matplotlib.pyplot as plt #!20220509
from matplotlib.figure import Figure   
from matplotlib.patches import Wedge
import imageio  #!20220520
import openmc
from utils.model import *
from utils.cal_param import *   #!20221023
from utils.move_detector import main
from utils.unet import *
from utils.jv_data import *

#%%
# load model, data
input_shape = 2  # [2, 5, 10, etc] (int) the size of the square detector. ['J', 'L', 'S', 'T', 'Z'] (string) for tetris detector.
seg_angles = 64 # segment of angles
file_header = '230118-005847_230120-214857' # save name of the model
model_path = f'./save/models/{file_header}_model.pt'
model =torch.load(model_path)
# RSID = np.array([[-4.0,11.0]]) # np.array([[1.0,2.0],[-3.0,14.0]])  # single source: np.array([[-4.0,11.0]]), double source: np.array([[1.0,2.0],[-3.0,14.0]])  #　The locations of radiation sources / an array with shape (n, 2)  (n: the number of radiation sources)
# rot_ratio = 0 # rotation ratio X, where \phi = X\theta
file_idx=6
jrad = 3
data_folder = './data/jayson/'
jvdict = load_jvdata(file_idx, data_folder)
recordpath = f'./save/mapping_data/jvdata{file_idx}_r{jrad}_v2.5'

#%%
# DT = 0.1
# SIM_TIME = 60.0
# STATE_SIZE = 4
# source_energies = [0.5e6 for _ in range(RSID.shape[0])]
SIM_STEP=1
ang_step_curves = True
round_digit=5
# num_particles = 50000
# sim_parameters = {
#     'DT': DT,
#     'SIM_TIME': SIM_TIME,
#     "STATE_SIZE": STATE_SIZE,
#     'RSID':RSID,
#     'source_energies':source_energies,
#     'SIM_STEP':SIM_STEP,
#     'rot_ratio': rot_ratio, 
#     'num_particles': num_particles
# }

# Map
# map_horiz = [2,12,20]
# map_vert = [-1,9,20]
map_horiz = [4,14,20]# [0,10,20]
map_vert = [-5,5,20]

colors_parameters = {'array_hex':'#EEAD0E', 'pred_hex':'#CA6C4A' , 'real_hex': '#77C0D2'}

#%%
# recordpath = './save/mapping_data/' + file_header
if __name__ == '__main__':  # and record_data:
    if not os.path.isdir(recordpath):
        os.mkdir(recordpath)
    os.system('rm ' + recordpath + "/*")
jsonpath = recordpath + "_json/"
if __name__ == '__maintribution__':  # and record_data:
    if not os.path.isdir(jsonpath):
        os.mkdir(jsonpath)
    os.system('rm ' + jsonpath + "*")
figurepath = recordpath + "_figure/"
if __name__ == '__main__':  # and record_data:
    if not os.path.isdir(figurepath):
        os.mkdir(figurepath)
    os.system('rm ' + figurepath + "*")
predictpath = recordpath + "_predicted/"
if __name__ == '__main__':  # and record_data:
    if not os.path.isdir(predictpath):
        os.mkdir(predictpath)
    os.system('rm ' + predictpath + "*")


if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")

DEFAULT_DTYPE = torch.double

#%%
def main_jv(recordpath, jvdict, seg_angles, model, colors_parameters, device):
    # DT = sim_parameters['DT']
    # SIM_TIME = sim_parameters['SIM_TIME']
    # STATE_SIZE = sim_parameters['STATE_SIZE']
    # RSID = sim_parameters['RSID']
    # source_energies = sim_parameters['source_energies']
    # SIM_STEP = sim_parameters['SIM_STEP']
    # rot_ratio = sim_parameters['rot_ratio']
    # num_particles = sim_parameters['num_particles']
    energy, times, det_id, jvdata, px, py, pz, qw, qx, qy, qz, x, y, z = [jvdict[k] for k in ['energy', 'timestamp', 'det_id', 'data', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'x', 'y', 'z']]
    pcoords = np.stack((px, py, pz), axis=-1)
    colors_max, pred_rgb, real_rgb = [hex2rgb(colors_parameters[l]) for l in ['array_hex', 'pred_hex', 'real_hex']]
    time=0
    step=0
    jsonpath = recordpath + "_json/"
    figurepath = recordpath + "_figure/"
    predictpath = recordpath + "_predicted/"
    num_panels=4
    matrix_shape = [2,2]
    relsrc = []
    relsrc_dir = []
    xTrue_record = []
    d_record=[]
    angle_record=[]
    predout_record=[]
    
    angles, axes = quaternions_to_rotations(qw, qx, qy, qz)
    # rotation axis
    zvecs = axes / np.linalg.norm(axes, axis=-1)[:, None]
    Zx, Zy, Zz = zvecs[:, 0], zvecs[:, 1], zvecs[:, 2]
    # dvecs = rotation_end_points(angles, axes)
    dvecs = rotation_end_points(angles+np.pi/2, axes)   #!
    yvecs = dvecs / np.linalg.norm(dvecs, axis=-1)[:, None]
    Yx, Yy, Yz = yvecs[:, 0], yvecs[:, 1], yvecs[:, 2]
    Y_reshaped = yvecs[:, :, None]
    Z_reshaped = zvecs[:, None, :]
    xvecs = np.cross(yvecs, zvecs)
    Xx, Xy, Xz = xvecs[:, 0], xvecs[:, 1], xvecs[:, 2]
    # for step, t in enumerate(times):  #!
    jall = range(jrad, len(times)-jrad)
    step = 0
    for j in jall:
        xTrue = np.array([px[step], py[step]]).reshape(-1, 1) #motion_model(xTrue, u, sim_parameters, rot_ratio=rot_ratio)
        # hxTrue: (4, ntimes). 4 rows: px, py, direction of X vec, direction of Y vec. 
        hxTrue = np.concatenate((pcoords[:step+1, :2].transpose(),  np.arctan2(Xy, Xx)[:step+1].reshape((1,-1)), np.arctan2(Yy, Yx)[:step+1].reshape((1,-1))), axis=0)

        det_output=None
        predict=None
        xTrue_record.append(xTrue.tolist())
        if step%SIM_STEP==0:
            print('STEP %d'%step)
            source_list=[]
            relsrc_line = []
            relsrc_dir_line = []
            sources_d_th = []
            sources_x_y_c = []
            angle_line = [step]

            '''
            The simulation function simulate the detector response given the radiation source position and intensity.
            The input of this function is a list of sources location (in detector frame, (0,0) is the center of the detector) and intensity eg, [[x1,y1,I1], [x2, y2, I2], ...]
            The output is an 1d array with shape (100,) that record the response of each single pad detector
            '''
            # signal = jvdata[step, :, :].sum(axis=0)
            jmin, jmax = j-jrad, j+jrad
            jrange = range(jmin, jmax+1)
            jnum = len(jrange)
            signal = np.zeros_like(jvdata[j, :, :].sum(axis=0))
            for j_ in jrange:
                signal += jvdata[j_, :, :].sum(axis=0)/jnum
            print([step, j], 'signal', signal)
            signal = signal.reshape((2,2), order='F')
            # print(signal)
            # signal = np.flip(signal, axis=0) 
            signal = np.flip(signal, axis=1)
            det_output = signal
            network_input = (det_output-det_output.mean())/np.sqrt(det_output.var())    # normalize
            print('network_input.shape: ', network_input.shape)
            print('network_input: ', network_input)
            network_input = np.transpose(network_input)
            print('network_input2.shape: ', network_input.shape)
            network_input = network_input.reshape(1,-1)
            print('network_input: ', network_input)
            network_input = torch.from_numpy(network_input).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
            predict=model(network_input.to(device)).detach().cpu().numpy().reshape(-1)

            xdata_original=det_output.reshape(matrix_shape)
            print('xdata_original.shape: ', xdata_original.shape)
            # ydata = get_output(sources_x_y_c, seg_angles)
            # pred_out = 180/math.pi*pipi_2_cw((2*math.pi/seg_angles)*(np.argmax(predict)-seg_angles/2))
            pred_out = 180/math.pi*pipi_2_cw((2*math.pi/seg_angles)*(calculate_expectation(np.arange(len(predict)), predict)-seg_angles/2))
            predout_record.append([step, pred_out])
            print("Result: " + str(pred_out) + " deg")
            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(42,20))
            xdata_show = np.flip(xdata_original, 0)
            xdata_show = np.flip(xdata_show, 1)
            adjust_ratio = 0.85
            matrix_len=xdata_show.flatten().shape[0]
            if matrix_len>num_panels:
                bottom_panel = np.partition(xdata_show.flatten(), -num_panels)[-num_panels]
                blank_panel = xdata_show.min()
                gap_pb = bottom_panel - blank_panel
                xdata_show = xdata_show - gap_pb*adjust_ratio*(xdata_show>=bottom_panel)
            else:
                bottom_panel = xdata_show.min()
                blank_panel = bottom_panel
                gap_pb = bottom_panel - blank_panel
            print(f'{num_panels}th largest', bottom_panel)
            print(f'{num_panels+1}th largest', blank_panel)
            print(gap_pb)
            print()

            N = 255
            rgbs = np.ones((N, 3))
            xx = max(colors_max)*(1-adjust_ratio)*((num_panels/matrix_len)==1)#70
            print('xx:', xx)
            rgbs[:, 0] = np.linspace(((colors_max[0]-255)/N*xx+255)/255, colors_max[0]/255, N) # R
            rgbs[:, 1] = np.linspace(((colors_max[1]-255)/N*xx+255)/255, colors_max[1]/255, N) # G
            rgbs[:, 2] = np.linspace(((colors_max[2]-255)/N*xx+255)/255, colors_max[2]/255, N)  # B
            own_cmp = ListedColormap(rgbs)
            ax1.imshow(xdata_show, interpolation='nearest', cmap=own_cmp)
            ags = angle_line
            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)
            plt.xlabel('x')
            plt.ylabel('y')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            theta = np.linspace(-90, 270, seg_angles)
            output1 = predict
            output2 = np.zeros_like(predict) #ydata.tolist()

            if ang_step_curves:
            #     ax2 = fig.add_subplot(122, polar=True)
                ax2 = plt.subplot(1, 2, 2, polar=True)
                theta_rad = np.linspace(180, -180, seg_angles) * np.pi/180
                ax2.plot(theta_rad,output2, drawstyle='steps', linestyle='-', color=rgb_to_hex(real_rgb), linewidth=7)  
                ax2.plot(theta_rad, output1, drawstyle='steps', linestyle='-', color=rgb_to_hex(pred_rgb), linewidth=7)
                ax2.set_yticklabels([])  # Hide radial tick labels
                ax2.tick_params(axis='x', labelsize=30)
                # Add the radial axis
                ax2.set_rticks(np.linspace(0, 1, 10))  # Adjust the range and number of radial ticks as needed
                ax2.spines['polar'].set_visible(True)  # Show the radial axis line

                # Set the theta direction to clockwise
                ax2.set_theta_direction(-1)
                # Set the theta zero location to the top
                ax2.set_theta_zero_location('N')
                ax2.set_rlabel_position(-22.5)
                ax2.set_theta_offset(np.pi / 2.0)
                ax2.tick_params(axis='x', which='major', pad=50, labelsize=40)
                ax2.grid(True)
                # ax2.set_frame_on(False)

            else:
                for i in range(len(theta)-1):
                    ax2.add_artist(
                        Wedge((0, 0), 1, theta[i], theta[i+1], width=0.3, 
                            color=((255-(255-pred_rgb[0])*output1[i])/255, 
                                    (255-(255-pred_rgb[1])*output1[i])/255, 
                                    (255-(255-pred_rgb[2])*output1[i])/255)),
                    )
                    ax2.add_artist(
                        Wedge((0, 0), 0.7, theta[i], theta[i+1], width=0.3, 
                            color=((255-(255-real_rgb[0])*output2[i])/255, 
                                    (255-(255-real_rgb[1])*output2[i])/255, 
                                    (255-(255-real_rgb[2])*output2[i])/255)),
                    )

                c1 = plt.Circle((0, 0), 1, color='k', lw=5, fill=False)
                c2 = plt.Circle((0, 0), 0.7, color='k', lw=5, fill=False)
                c3 = plt.Circle((0, 0), 0.4, color='k', lw=5, fill=False)
                ax2.add_patch(c1)
                ax2.add_patch(c2)
                ax2.add_patch(c3)
                ax2.set_xlim((-1.2,1.2))
                ax2.set_ylim((-1.2,1.2))
                ax2.axes.get_xaxis().set_visible(False)
                ax2.axes.get_yaxis().set_visible(False)
                
            # fig.suptitle('Real Angle: ' + str(round(ags[-1], 4)) + ', \nPredicted Angle: ' + str(pred_out) + ' [deg]', fontsize=60)
            fig.suptitle('Real Angle: ' + str(round(ags[-1], round_digit)) + ', \nPredicted Angle: ' + str(round(pred_out, round_digit)) + ' [deg]', fontsize=60)
            fig.savefig(predictpath + 'STEP%.4d'%step + "_predict.png")
            fig.savefig(predictpath + 'STEP%.4d'%step + "_predict.pdf")
            plt.close(fig)
            print(step,'simulation end')
            
            
        if record_data:
            data_dump={
            # 'RSID':RSID,
            'hxTrue':hxTrue,
            'det_output':det_output,
            'predict_list':predict
            }
            with open(os.path.join(recordpath,'STEP%.4d.pkl'%step),'wb') as f:
                pkl.dump(data_dump,f)
            try:
                with open(os.path.join(recordpath,'STEP%.4d.pkl'%step),'rb') as f:
                    data=pkl.load(f, encoding="latin1")
                print('File pickle success: ' + 'STEP%.4d.pkl'%step)
            except EOFError:
                print('EOFError: ' + 'STEP%.4d.pkl'%step)
        
        step += 1
    
    xTrue_data = np.array(xTrue_record)
    plt.plot(xTrue_data[:,0], xTrue_data[:,1])
    plt.title('xTrue: robot trajectory')
    plt.savefig(figurepath + 'xTrue.png')
    plt.savefig(figurepath + 'xTrue.pdf')
    plt.close()


    with imageio.get_writer(figurepath+'move_detector.gif', mode='I') as writer:
        for figurename in sorted(os.listdir(predictpath)):
            if figurename.endswith('png'):
                image = imageio.imread(predictpath + "/" +figurename)
                writer.append_data(image)

#%%
# main(recordpath, tetris_mode, input_shape, seg_angles, model, sim_parameters, colors_parameters, device=DEFAULT_DEVICE)
# write_data(seg_angles, recordpath, map_horiz, map_vert)
# main(recordpath, tetris_mode, input_shape, seg_angles, model, sim_parameters, colors_parameters, device=DEFAULT_DEVICE)
main_jv(recordpath, jvdict, seg_angles, model, colors_parameters, device=DEFAULT_DEVICE)
write_data(seg_angles, recordpath, map_horiz, map_vert)

#%%
