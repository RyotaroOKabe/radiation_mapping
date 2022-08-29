#%%

"""
2022/08/15
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys,os
import pickle as pkl

#sys.path.append('../')
sys.path.append('./')   #!20220331
from train_v1 import *  #!20220717
#?from train_torch_openmc_tetris_v1 import *  #!20220717

import matplotlib.pyplot as plt #!20220509
from matplotlib.figure import Figure   
from matplotlib.patches import Wedge
import imageio  #!20220520
import openmc
from mcsimulation_square import *
from cal_param import *

a_num =2
num_sources = 1
seg_angles = 64
file_header = f"A20220822_{a_num}x{a_num}_{num_sources}src_{seg_angles}_v1.1"
recordpath = f'mapping_data/mapping_{file_header}'
model_path = f'save_model/model_openmc_{a_num}x{a_num}_{num_sources}src_{seg_angles}_ep500_bs256_20220822_v1.1_model.pt'
model =torch.load(model_path)

DT = 0.1  # time tick [s]
SIM_TIME = 70.0
STATE_SIZE = 3
RSID = np.array([[-7.0,10.5,0.5e6]]) #np.array([[1.0,2.0,0.5e6],[-3.0,14.0,0.5e6]])  
source_energies = [0.5e6, 0.5e6]
SIM_STEP=10
sim_parameters = {
    'DT': DT,
    'SIM_TIME': SIM_TIME,
    "STATE_SIZE": STATE_SIZE,
    'RSID':RSID,
    'source_energies':source_energies,
    'SIM_STEP':SIM_STEP,
}

# Map
map_horiz = [-15,15,30]
map_vert = [-5,25,30]

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


if torch.cuda.is_available() and not USE_CPU:
    DEFAULT_DEVICE = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEFAULT_DEVICE = torch.device("cpu")

DEFAULT_DTYPE = torch.double

#%%
def openmc_simulation(sources_d_th, header, seg_angles):  #!20220717
    num_particles = 50000
    num_sources = len(sources_d_th)
    for i in range(num_sources):
        sources_d_th[i][0] *= 100
    before_openmc(a_num, sources_d_th, num_particles, seg_angles)  #!20220803
    openmc.run()
    mm = after_openmc(a_num, sources_d_th, jsonpath, figurepath, seg_angles, header) 
    return mm




def main(seg_angles, model, recordpath, sim_parameters):
    DT = sim_parameters['DT']
    SIM_TIME = sim_parameters['SIM_TIME']
    STATE_SIZE = sim_parameters['STATE_SIZE']
    RSID = sim_parameters['RSID']
    source_energies = sim_parameters['source_energies']
    SIM_STEP = sim_parameters['SIM_STEP']
    
    time=0
    step=0
    

    relsrc = []
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

        xTrue = motion_model(xTrue, u, sim_parameters)


        hxTrue = np.hstack((hxTrue, xTrue))

        det_output=None
        predict=None

        xTrue_record.append(xTrue)

        if step%SIM_STEP==0:
            print('STEP %d'%step)
            source_list=[]
            relsrc_line = []
            d_record_line = [step]
            angle_line = [step]
            dist_ang_list = []
            sources_d_th = []
            sources_x_y_c = []
            for i in range(RSID.shape[0]):
                dx = RSID[i, 0] - xTrue[0, 0]
                dy = RSID[i, 1] - xTrue[1, 0]
                d = math.sqrt(dx**2 + dy**2)
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
                
                src_xy = [x, y]
                source_x_y_c = {}
                source_x_y_c['position']=src_xy
                source_x_y_c['counts']=source_energies[i]
                sources_x_y_c.append(source_x_y_c)
                
                relsrc_line.append(x)
                relsrc_line.append(y)
                
                d_record_line.append(d)
                angle_line.append(angle*180/np.pi)
                dist_ang_list.append([d, angle*180/np.pi])
                sources_d_th.append([d, angle*180/np.pi, source_energies[i]])

            relsrc.append(relsrc_line)
            d_record.append(d_record_line)
            angle_record.append(angle_line)
            dist_ang = np.transpose(np.array(dist_ang_list))


            print(step,'simulation start')
            print("source")
            print(source_list)

            det_output=openmc_simulation(sources_d_th, 'STEP%.3d'%step, seg_angles)
            print('det_output')
            print(type(det_output))
            print(det_output.shape)
            '''
            The simulation function simulate the detector response given the radiation source position and intensity.
            The input of this function is a list of sources location (in detector frame, (0,0) is the center of the detector) and intensity eg, [[x1,y1,I1], [x2, y2, I2], ...]
            The output is an 1d array with shape (100,) that record the response of each single pad detector
            '''
            network_input = (det_output-det_output.mean())/np.sqrt(det_output.var())
            network_input = np.transpose(network_input)
            network_input = network_input.reshape(1,-1)

            network_input = torch.from_numpy(network_input).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

            predict=model(network_input).detach().cpu().numpy().reshape(-1)

            xdata_original=det_output.reshape(a_num, a_num)
            ydata = get_output_mul(sources_x_y_c, seg_angles)
            pred_out = (360/seg_angles)*(np.argmax(predict)-seg_angles/2)
            predout_record.append([step, pred_out])
            print("Result: " + str(pred_out) + " deg")

            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(40,20))    #!20220520

            xdata_show = np.flip(xdata_original, 0)
            xdata_show = np.flip(xdata_show, 1)
            ax1.imshow(xdata_show, interpolation='nearest', cmap="YlOrBr")
            ds, ag = d, angle*180/np.pi
            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)
            plt.xlabel('x')
            plt.ylabel('y')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')

            theta = np.linspace(-90, 270, seg_angles)   #!20220729
            output1 = predict
            output2 = ydata.tolist()

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

            ax2.set_xlim((-1.2,1.2))
            ax2.set_ylim((-1.2,1.2))
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)

            fig.suptitle('Real Angle: ' + str(round(360-ag, 5)) + ', \nPredicted Angle: ' + str(360-pred_out) + ' [deg]', fontsize=60)  ##!20220822
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

    with imageio.get_writer('mapping_data/save_fig/'+file_header + '.gif', mode='I') as writer:
        for figurename in sorted(os.listdir(predictpath)):
            if figurename.endswith('png'):  #!20220729
                image = imageio.imread(predictpath + "/" +figurename)
                writer.append_data(image)
    pass



#%%
if __name__ == '__main__':
    main(seg_angles, model, recordpath, sim_parameters)
    write_data(seg_angles, recordpath, map_horiz, map_vert)

