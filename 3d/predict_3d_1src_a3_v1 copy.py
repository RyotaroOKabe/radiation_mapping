#%%
import numpy as np
import math
import matplotlib.pyplot as plt
import sys, os
import pickle as pkl


#sys.path.append('../')
sys.path.append('./')   #!20220331
#from train_torch_openmc import *    #!20220502 out!
from train_torch_openmc_a3_v1 import *    #!20220502 out!
from gen_openmc_data_discrete_a3_v2 import *

from cal_param_a3_v1 import *
from dataset_a3_v1 import *
import torch
import copy

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


predictpath = 'predict_test/20220704_predict_10cmxx3_v1.1'
isExist1 = os.path.exists(predictpath)
if not isExist1:
# Create a new directory because it does not exist 
    os.makedirs(predictpath)
    print("The new directory "+ predictpath +" is created!")
#os.remove(predictpath + "*.png")

model_path = 'save_model/model_20220704_openmc_10cm_10xx3_ep100_bs256_v1.1_model.pt' #!20220502 out
model =torch.load(model_path)
print(model)    #!20220412

#data_path = 'openmc/data_20220226_1.2'  #!20220502 out
data_path = 'openmc/discrete_data_20220630_10^3_v1'  #!20220502


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi



def plot_angle_matrices_pred(data1, output, filename, filterpath, filter_types_dict, a_num): #!20220701
    
    phi_list, theta_list = angles_lists(filterpath=filterpath)
    file_data_4d = tensor_types_angles_data(filter_types_dict, phi_list, theta_list, a_num)
    filter_types = list(filter_types_dict.keys())

    data1_input = data1['input']
    data1_mtrx = data1['output_mtrx']
    data1_output = np.einsum('ijkl,l->ijk', file_data_4d, data1_input)
    fig = plt.figure(figsize=(20, 14))

    dm = np.array(data1_mtrx)
    id_t_m, id_p_m = [int(np.where(dm==dm.max())[i]) for i in range(2)]
    ax1 = fig.add_subplot(411)
    ax1.imshow(dm, extent=[phi_list[0],phi_list[-1],theta_list[-1],theta_list[0]])
    ax1.set_title(f"[{filename[:-5]}]\nPeak: \u03C6 = {phi_list[id_p_m]} deg ({id_p_m}) /  \u03B8 = {theta_list[id_t_m]} deg ({id_t_m})")

    do0 = output
    id_t_0, id_p_0 = [int(np.where(do0==do0.max())[i]) for i in range(2)]
    ax2 = fig.add_subplot(412)
    ax2.imshow(do0, extent=[phi_list[0],phi_list[-1],theta_list[-1],theta_list[0]])
    ax2.set_title(f"[Prediction]\nPeak: \u03C6 = {phi_list[id_p_0]} deg ({id_p_0}) /  \u03B8 = {theta_list[id_t_0]} deg ({id_t_0})")
    ax2.set_ylabel('\u03B8 [deg]', fontsize = 16)

    fig.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.4)
    


def main(a_num):

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
            #ydata=get_output(data['source'])
            ydata = data['output_mtrx'] #!20220704
            ydata_ph, ydata_th=get_output(data['source'])
            ydata=np.array(ydata)

        #?+++++++++++++++++
                #predict=model(network_input).detach().cpu().numpy().reshape(-1)
                #predict=model(network_input)#.detach().cpu().numpy().reshape(-1)
        #xdata.to(DEFAULT_DEVICE)
        #xdata.to(torch.device("cpu"))
        #xdata.to("cuda")
        print('```````````````````````')
        print("prediction starts here: " + filename)
        print(xdata.shape)
        xdata=xdata.reshape(1, -1)  #!20220704
        print(xdata.shape)
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
        #3predict=predict.reshape(-1)
        predict=predict.reshape(ydata_th.shape[0], ydata_ph.shape[0])
        print('""""""""predict.reshape(-1)""""""""""')
        print(predict)
            #predict=model(network_input).detach().cpu().numpy()#.reshape(-1)
        #?+++++++++++++++++

            #print(step,'simulation end')
        print('+++++++++++++++++++++++++')
        print("prediction complete: " + filename)
        #print(predict.shape)
        #print(type(predict))
        print(predict)
        pred_out = 9*(np.argmax(predict)-20)
        #print("Result: " + str(pred_out) + " deg")
        
        #xdata_original=xdata_original.reshape(10, 10)
        xdata_original=xdata_original.reshape(a_num, a_num, a_num)   #!20220704
        
        
        fig, axs = plt.subplots(3, a_num, figsize=(a_num*5,15), constrained_layout=True)  #!20220629 #!a_num
        fs_label = 20
        fs_title = 22
        fs_tick = 18
        
        mean = xdata_original
        maxmax = mean.max()
        minmin = mean.min()
        
        ds, ph, th = filename[:-5].split('_')[1:]   #!20220704
        
        for xa in range(a_num): #!20220629 #!a_num
            ax = axs[0, xa]
            #axes.append(fig.add_subplot(3, 5, xa+1) )
            xslice = ax.imshow(mean[xa, :, :], vmin=minmin, vmax=maxmax, interpolation='nearest', cmap="plasma")       #!20220118
            ax.set_xlabel('z', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
            ax.set_ylabel('y', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
            ax.tick_params(axis='x', labelsize=fs_tick)
            ax.tick_params(axis='y', labelsize=fs_tick)
            #plt.colorbar()
            ax.set_title("X=" + str(xa), fontsize = fs_title)
        for ya in range(a_num):
            ax = axs[1, ya]
            #axes.append(fig.add_subplot(3, 5, ya+6) )
            yslice = ax.imshow(mean[:, ya, :], vmin=minmin, vmax=maxmax, interpolation='nearest', cmap="plasma")       #!20220118
            ax.set_xlabel('z', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
            ax.set_ylabel('x', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
            ax.tick_params(axis='x', labelsize=fs_tick)
            ax.tick_params(axis='y', labelsize=fs_tick)
            #plt.colorbar()
            ax.set_title("Y=" + str(ya), fontsize = fs_title)
        for za in range(a_num):
            ax = axs[2, za]
            #axes.append(fig.add_subplot(3, 5, za+11) )
            zslice = ax.imshow(mean[:, :, za], vmin=minmin, vmax=maxmax, interpolation='nearest', cmap="plasma")       #!20220118
            ax.set_xlabel('y', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
            ax.set_ylabel('x', fontsize = fs_label)  #!20220502 Adjust the incorrect axis setting!
            ax.tick_params(axis='x', labelsize=fs_tick)
            ax.tick_params(axis='y', labelsize=fs_tick)
            #plt.colorbar()
            ax.set_title("Z=" + str(za), fontsize = fs_title)
        
        fig.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.8, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.3)
    
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(zslice, ax=axs[:, -1], cax=cbar_ax)#, location='right', cax=cbar_ax)#, shrink=0.6)
        #fig.colorbar(zslice, ax=axs[:, -1], location='right')#, shrink=0.6)
        #fig.colorbar(zslice, location='right')#, shrink=0.6)
        #?fig.suptitle('dist: ' + ds + ',  \u03C6: ' + ph + ',  \u03B8: ' + th + '\nMean_max: ' + str(max) + '\nStdev_max: ' + str(stdev_max), fontsize=25)
        fig.suptitle('[real] r: ' + ds + ' /  \u03C6: ' + ph + ' /  \u03B8: ' + th + "\n"
                     '[Pred] r: ' + ds + ' /  \u03C6: ' + ph + ' /  \u03B8: ' + th, fontsize=30)
        fig.savefig(predictpath + '/' + filename[:-4] + "png") #   'random_savefig/abs_rate_20220118_6.png')   #!20220117
        plt.close()
    
        print('[real] r: ' + ds + ' /  \u03C6: ' + ph + ' /  \u03B8: ' + th + "\n"
                     '[Pred] r: ' + ds + ' /  \u03C6: ' + ph + ' /  \u03B8: ' + th)
        
        
        
        #?xdata_original=np.transpose(xdata_original)
        #print(xdata_original)
        #print(xdata_original.shape)
        #plt.imshow(xdata_original, interpolation='nearest')  #, cmap="plasma")
        #?ds, ph, th = filename[:-5].split('_')[1:]   #!20220704
        #plt.title('R_dist: ' + ds + ',  R_angle: ' + ag + '\nP_angle: ' + str(pred_out))   #Mean_max: ' + str(max) + '\nStdev_max: ' + str(stdev_max))
        #plt.xlabel('x')    #!20220502 out
        #plt.ylabel('y')    #!20220502 out
        #plt.xlabel('y')    #!20220502 
        #plt.ylabel('x')    #!20220502 
        #plt.colorbar()
        #plt.savefig(predictpath + '/' + filename[:-4] + "png")
        #plt.close()

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
    main(a_num=10)



# %%
