#%%
import numpy as np
import math
import matplotlib.pyplot as plt
import sys,os
import pickle as pkl
sys.path.append('./')
from utils.model import *
from utils.utils import *
import matplotlib.pyplot as plt
from matplotlib.figure import Figure   
from matplotlib.patches import Wedge
from matplotlib.colors import ListedColormap
import imageio
import openmc
from utils.cal_param import *
if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.double
ang_step_curves = True   #!

#%%
N = 256

def openmc_simulation(tetris_mode, input, sources_d_th, header, seg_angles, num_particles, jsonpath):
    num_sources = len(sources_d_th)
    for i in range(num_sources):
        sources_d_th[i][0] *= 100
    if tetris_mode:
        from utils.mcsimulation_tetris import get_tetris_shape, before_openmc, after_openmc
        shape_name = input
        use_panels = get_tetris_shape(shape_name)
        before_openmc(use_panels, sources_d_th, num_particles)
        openmc.run()
        mm = after_openmc(use_panels, sources_d_th, jsonpath, seg_angles, header) 
    else:
        from utils.mcsimulation_square import before_openmc, after_openmc
        a_num = input
        before_openmc(a_num, sources_d_th, num_particles)
        openmc.run()
        mm = after_openmc(a_num, sources_d_th, jsonpath, seg_angles, header) 
    return mm

def main(recordpath, tetris_mode, input, seg_angles, model, sim_parameters, colors_parameters, device):
    DT = sim_parameters['DT']
    SIM_TIME = sim_parameters['SIM_TIME']
    STATE_SIZE = sim_parameters['STATE_SIZE']
    RSID = sim_parameters['RSID']
    source_energies = sim_parameters['source_energies']
    SIM_STEP = sim_parameters['SIM_STEP']
    rot_ratio = sim_parameters['rot_ratio']
    num_particles = sim_parameters['num_particles']
    colors_max, pred_rgb, real_rgb = [hex2rgb(colors_parameters[l]) for l in ['array_hex', 'pred_hex', 'real_hex']]
    time=0
    step=0
    jsonpath = recordpath + "_json/"
    figurepath = recordpath + "_figure/"
    predictpath = recordpath + "_predicted/"
    if tetris_mode:
        num_panels=4
        matrix_shape = [2,3]
    else:
        num_panels=input**2
        matrix_shape = [input, input]
    relsrc = []
    relsrc_dir = []
    xTrue_record = []
    d_record=[]
    angle_record=[]
    predout_record=[]
    xTrue = np.zeros((STATE_SIZE, 1))
    hxTrue = xTrue

    while SIM_TIME >= time:
        time += DT
        step+=1
        u = calc_input(time)
        xTrue = motion_model(xTrue, u, sim_parameters, rot_ratio=rot_ratio)
        hxTrue = np.hstack((hxTrue, xTrue))
        det_output=None
        predict=None
        xTrue_record.append(xTrue.tolist())
        if step%SIM_STEP==0:
            print('STEP %d'%step)
            source_list=[]
            relsrc_line = []
            relsrc_dir_line = []
            d_record_line = [step]
            angle_line = [step]
            sources_d_th = []
            sources_x_y_c = []
            for i in range(RSID.shape[0]):
                dx = RSID[i, 0] - xTrue[0, 0]
                dy = RSID[i, 1] - xTrue[1, 0]
                d = math.sqrt(dx**2 + dy**2)
                angle = pi_2_pi(math.atan2(dy, dx) - xTrue[3, 0])
                angle_dir = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
                print('RSID:', [RSID[i, 0],RSID[i, 1]])
                print('xTrue:', [xTrue[0, 0],xTrue[1, 0]])
                print('d:', d)
                print('[dx, dy]:', [dx, dy])
                print('angle (rad): ', angle)
                print('angle (deg): ', angle*180/np.pi)
                print('xTrue_ang: ', xTrue[3, 0])
                x=d*np.cos(angle)
                y=d*np.sin(angle)
                x_dir=d*np.cos(angle_dir)
                y_dir=d*np.sin(angle_dir)
                source_list.append([x,y,source_energies[i]])
                src_xy = [x, y]
                source_x_y_c = {}
                source_x_y_c['position']=src_xy
                source_x_y_c['counts']=source_energies[i]
                sources_x_y_c.append(source_x_y_c)
                relsrc_line.append(x)
                relsrc_line.append(y)
                relsrc_dir_line.append(x_dir)
                relsrc_dir_line.append(y_dir)
                d_record_line.append(d)
                angle_line.append(pipi_2_cw(angle)*180/math.pi)
                sources_d_th.append([d, angle*180/np.pi, source_energies[i]])

            relsrc.append(relsrc_line)
            relsrc_dir.append(relsrc_dir_line)
            d_record.append(d_record_line)
            angle_record.append(angle_line)

            print(step,'simulation start')
            print("source")
            print(source_list)
            det_output=openmc_simulation(tetris_mode, input, sources_d_th, 'STEP%.3d'%step, seg_angles, num_particles, jsonpath)
            print('det_output')
            print(type(det_output))
            print(det_output.shape)
            '''
            The simulation function simulate the detector response given the radiation source position and intensity.
            The input of this function is a list of sources location (in detector frame, (0,0) is the center of the detector) and intensity eg, [[x1,y1,I1], [x2, y2, I2], ...]
            The output is an 1d array with shape (100,) that record the response of each single pad detector
            '''
            network_input = (det_output-det_output.mean())/np.sqrt(det_output.var())    # normalize
            network_input = np.transpose(network_input)
            network_input = network_input.reshape(1,-1)
            network_input = torch.from_numpy(network_input).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
            predict=model(network_input.to(device)).detach().cpu().numpy().reshape(-1)

            xdata_original=det_output.reshape(matrix_shape)
            ydata = get_output(sources_x_y_c, seg_angles)
            pred_out = 180/math.pi*pipi_2_cw((2*math.pi/seg_angles)*(np.argmax(predict)-seg_angles/2))
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
            theta = np.linspace(-90, 270, seg_angles)   #!20220729
            output1 = predict
            output2 = ydata.tolist()

            if ang_step_curves: #!
            #     ax2 = fig.add_subplot(122, polar=True)
                ax2 = plt.subplot(1, 2, 2, polar=True)
                theta_rad = theta * np.pi/180
                ax2.plot(theta_rad, output1, drawstyle='steps', linestyle='-', color=rgb_to_hex(pred_rgb), linewidth=7)
                ax2.plot(theta_rad,output2, drawstyle='steps', linestyle='-', color=rgb_to_hex(real_rgb), linewidth=7)  
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
                
            fig.suptitle('Real Angle: ' + str(round(ags[-1], 4)) + ', \nPredicted Angle: ' + str(pred_out) + ' [deg]', fontsize=60)
            fig.savefig(predictpath + 'STEP%.3d'%step + "_predict.png")
            fig.savefig(predictpath + 'STEP%.3d'%step + "_predict.pdf")
            plt.close(fig)
            print(step,'simulation end')
            
            
        if record_data:
            data_dump={
            'RSID':RSID,
            'hxTrue':hxTrue,
            'det_output':det_output,
            'predict_list':predict
            }
            with open(os.path.join(recordpath,'STEP%.3d.pkl'%step),'wb') as f:
                pkl.dump(data_dump,f)
            try:
                with open(os.path.join(recordpath,'STEP%.3d.pkl'%step),'rb') as f:
                    data=pkl.load(f, encoding="latin1")
                print('File pickle success: ' + 'STEP%.3d.pkl'%step)
            except EOFError:
                print('EOFError: ' + 'STEP%.3d.pkl'%step)

    rel_source = np.array(relsrc)
    plt.plot(rel_source[:,0], rel_source[:,1])
    plt.title('relative source position')
    plt.savefig(figurepath + 'rel_source.png')
    plt.savefig(figurepath + 'rel_source.pdf')
    plt.close()

    rel_dir_source = np.array(relsrc_dir)
    plt.plot(rel_dir_source[:,0], rel_dir_source[:,1])
    plt.title('relative source position')
    plt.savefig(figurepath + 'rel_dir_source.png')
    plt.savefig(figurepath + 'rel_dir_source.pdf')
    plt.close()
    
    xTrue_data = np.array(xTrue_record)
    plt.plot(xTrue_data[:,0], xTrue_data[:,1])
    plt.title('xTrue: robot trajectory')
    plt.savefig(figurepath + 'xTrue.png')
    plt.savefig(figurepath + 'xTrue.pdf')
    plt.close()
    
    d_data = np.array(d_record)
    plt.plot(d_data[:,0], d_data[:,1])
    plt.title('distance')
    plt.xlabel('step')
    plt.ylabel('distance [m]')
    plt.savefig(figurepath + 'dist.png')
    plt.savefig(figurepath + 'dist.pdf')
    plt.close()

    angle_data = np.array(angle_record)
    predout_data = np.array(predout_record)
    plt.plot(DT*angle_data[:,0], angle_data[:,1], label="angle", color='#64ADB1')
    plt.plot(DT*predout_data[:,0], predout_data[:,1], label="pred_out", color='#D58B70')
    plt.title('angle')
    plt.xlabel('Time [s]')
    plt.ylabel('angle [deg]')
    plt.legend(loc="upper right")
    plt.ylim(0,360)
    plt.savefig(figurepath + 'angle.png')
    plt.savefig(figurepath + 'angle.pdf')
    plt.close()

    with imageio.get_writer(figurepath+'move_detector.gif', mode='I') as writer:
        for figurename in sorted(os.listdir(predictpath)):
            if figurename.endswith('png'):
                image = imageio.imread(predictpath + "/" +figurename)
                writer.append_data(image)
    pass


