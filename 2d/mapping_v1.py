#%%
# -*- coding: utf-8 -*-
import numpy as np

from pydrake.all import MathematicalProgram, Solve

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as matplotlib_polygon
import matplotlib
import math

from pydrake.math import sin, cos, sqrt

import pickle as pkl

from scipy.interpolate import interp2d

import os,sys
import dill #!20220316
import imageio

a_num = 10
num_sources = 2
seg_angles = 128
fig_folder = f'mapping_data/save_fig/'
fig_header = f'A20220813_{a_num}x{a_num}_{num_sources}src_{seg_angles}_v1.1.1'
record_path = f'mapping_data/mapping_A20220813_{a_num}x{a_num}_{num_sources}src_{seg_angles}_v1.1'   #'mapping_data/mapping_A20220804_10x10_v1.7'
save_process = True
savedata=True
factor1 = 1e+24
# Map
map_horiz = [-15,15,30]
map_vert = [-5,25,30]


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
        self.x_list=np.linspace(x_min+self.dx/2,x_max-self.dx/2,self.x_num)
        self.y_list=np.linspace(y_min+self.dy/2,y_max-self.dy/2,self.y_num)
        self.center_list=[]
        self.square_list=[]
        for i in range(self.x_num):
            for j in range(self.y_num):  
                 x= x_min + self.dx/2 + i * self.dx
                 y= y_min + self.dy/2 + j * self.dy
                 self.center_list.append([x,y])
        self.intensity_list=np.zeros(self.x_num*self.y_num)
        self.intensity=self.intensity_list.reshape((self.x_num,self.y_num))

    def plot(self,ax=plt.gca()):
        X, Y = np.meshgrid(self.x_list, self.y_list)
        plt.figure(figsize=(8, 8))  #!20220520
        cmap=matplotlib.cm.get_cmap('Reds')
        plt.pcolormesh(X, Y, self.intensity, cmap=cmap, shading='gouraud')
        plt.savefig(fname=fig_folder+'/'+fig_header+"_plot1.png") #!20220323
        plt.savefig(fname=fig_folder+'/'+fig_header+"_plot1.pdf") #!20220323
        print("Plot1")

def solve_one(m,cji_list,yj_list):
    mp = MathematicalProgram()

    xi=mp.NewContinuousVariables(m.size, "xi")

    for i in range(m.size):
        mp.AddLinearConstraint(xi[i] >= 0.)

    reg=0.1
    mp.AddQuadraticCost(xi.dot(xi)*reg)

    for i in range(len(cji_list)):

        cji=cji_list[i]
        yj=yj_list[i]

        dd=cji.dot(xi)-yj
        mp.AddQuadraticCost(dd.dot(dd))



    result = Solve(mp)
    print(result.is_success())

    x=result.GetSolution(xi)

    return x


def main():
    recordpath = record_path+'_cal'   #!20220331
    files=os.listdir(recordpath)
    files=sorted(files)
    figurepath = 'mapping_data/save_fig/' + fig_header
    figurepath_pdf = figurepath + '_pdf'
    if save_process:
        if not os.path.isdir(figurepath):
            os.mkdir( figurepath)
        if not os.path.isdir(figurepath_pdf):
            os.mkdir(figurepath_pdf)
    os.system('rm ' +  figurepath + "/*")    #!20220509
    os.system('rm ' +  figurepath_pdf + "/*")    #!20220509
    m=Map(map_horiz, map_vert)   #!20220516 

    cji_list=[]
    yj_list=[]

    for filename in files:

        try:
            with open(os.path.join(recordpath,filename),'rb') as f:
                data=pkl.load(f, encoding="latin1")
        except EOFError:
            print('EOFError when loading : ' + recordpath+'/'+filename)

        if data['det_output'] is None: #!20220322 (tentative)
            continue

        cji=data['cji']
        yj=data['yj'].reshape(-1)
        yj_new = abs(yj)*factor1    #!20220515

        cji_list.append(cji)
        yj_list.append(yj_new)

        RSID=data['RSID']

        x=solve_one(m,cji_list,yj_list)
        print(filename)
        print(x)

        if savedata:
            data['xi']=x
            if not os.path.isdir(recordpath+'_final'):    #!20220510
                os.mkdir(recordpath+'_final')
            try:    #!20220510
                with open(os.path.join(recordpath+'_final',filename),'wb') as f:
                    pkl.dump(data,f)
            except EOFError:
                print('EOFError dumping: ' + recordpath+'_final/'+filename)

        if save_process:
            m.intensity=x.reshape(m.x_num,m.y_num).T
            hxTrue_data = np.array(data['hxTrue'])
            pose_x = 10*(hxTrue_data[0,-1] - hxTrue_data[0,-2])
            pose_y = 10*(hxTrue_data[1,-1] - hxTrue_data[1,-2])
            m.plot()
            for i in range(RSID.shape[0]):
                plt.plot(RSID[i, 0],RSID[i, 1],"xk")  #!20220804 multi sources
            plt.plot(hxTrue_data[0,:], hxTrue_data[1, :], linewidth=2)
            plt.arrow(hxTrue_data[0,-2], hxTrue_data[1, -2], pose_x, pose_y, head_width = 0.8, width=0.1)
            plt.plot(hxTrue_data[0,-1], hxTrue_data[1, -1],"o", color='blue', markersize=7)
            plt.title('STEP: ' + filename[4:7], fontsize=20)
            plt.savefig(fname=figurepath+'/'+filename[:7] +".png") #!20220323
            plt.savefig(fname=figurepath_pdf+'/'+filename[:7] +".pdf") #!20220813
            plt.show()

    m.intensity=x.reshape(m.x_num,m.y_num).T
    m.plot()
    for i in range(RSID.shape[0]):
        plt.plot(RSID[i, 0],RSID[i, 1],"xk")
    plt.savefig(fname=fig_folder+'/'+fig_header+"_plot2.png")
    plt.show()
    print("Plot2")

    with imageio.get_writer(fig_folder+'/'+fig_header+'.gif', mode='I') as writer:
        for figurename in sorted(os.listdir(figurepath)):
            image = imageio.imread(figurepath + '/' + figurename)
            writer.append_data(image)
    pass

def test():
    m=Map(map_horiz, map_vert)
    m.intensity=np.random.rand(m.x_num,m.y_num).T
    m.plot()
    plt.show()
    plt.savefig(fname=fig_folder+'/'+fig_header+"_plot3.png")
    plt.savefig(fname=fig_folder+'/'+fig_header+"_plot3.pdf")

def gen_gif():
    figurepath = 'mapping_data/save_fig/' + fig_header
    with imageio.get_writer('mapping_data/save_fig/'+fig_header+'.gif', mode='I') as writer:
        for figurename in sorted(os.listdir(figurepath)):
            image = imageio.imread(figurepath + '/' + figurename)
            writer.append_data(image)
    print("Finish making a gif: " + 'mapping_data/save_fig/'+fig_header+'.gif')

    
#%%
if __name__ == '__main__':
    main()
    test()
    gen_gif()
        
# %%
