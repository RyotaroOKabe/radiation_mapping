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

# a_num = 2
# num_sources = 1
# seg_angles = 64
# fig_folder = f'mapping_data/save_fig/'
# fig_header = f'A20221024_{a_num}x{a_num}_{num_sources}src_{seg_angles}_v1.13.1'
# record_path = f'mapping_data/mapping_A20221024_{a_num}x{a_num}_{num_sources}src_{seg_angles}_v1.13'   #'mapping_data/mapping_A20220804_10x10_v1.7'
# th_level = 0.2
# save_process = True
# savedata=True
factor1 = 1e+24

# a_num = 5
# num_sources = 2
# seg_angles = 64
# fig_folder = f'mapping_data/save_fig/'
# fig_header = f'A20221024_{a_num}x{a_num}_{num_sources}src_{seg_angles}_v3.5.1'
# record_path = f'mapping_data/mapping_A20221024_{a_num}x{a_num}_{num_sources}src_{seg_angles}_v3.5'   #'mapping_data/mapping_A20220804_10x10_v1.7'
th_level = 0.2
save_process = True
savedata=True

# Map
# map_horiz = [-15,15,30]
# map_vert = [-5,25,30]


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

    def normalize(self):    #!
        self.intensity = self.intensity/np.max(self.intensity)  #!
        
    def threshold(self, threshold): #!
        # self.intensity = np.maximum(self.intensity, threshold)  #!
        mask = self.intensity>threshold
        self.intensity = np.maximum(self.intensity-threshold, 0) + threshold*mask #!

    # def plot(self,ax=plt.gca()):
    def plot(self,ax=plt.gca(), fig_folder='XX', fig_header='XX'):
        X, Y = np.meshgrid(self.x_list, self.y_list)
        plt.figure(figsize=(8, 8))  #!20220520
        cmap=matplotlib.cm.get_cmap('Reds')
        plt.pcolormesh(X, Y, self.intensity, cmap=cmap, shading='gouraud')
        # plt.savefig(fname=fig_folder+'/'+fig_header+"_plot1.png") #!20220323
        # plt.savefig(fname=fig_folder+'/'+fig_header+"_plot1.pdf") #!20220323
        # print("Plot1")

# def solve_one(m,cji_list,yj_list):
#     mp = MathematicalProgram()  #!!!!
#     xi=mp.NewContinuousVariables(m.size, "xi")  #!!!!
#     for i in range(m.size):
#         mp.AddLinearConstraint(xi[i] >= 0.)  #!!!!
#     reg=0.1
#     mp.AddQuadraticCost(xi.dot(xi)*reg)  #!!!!
#     for i in range(len(cji_list)):
#         cji=cji_list[i]
#         yj=yj_list[i]
#         dd=cji.dot(xi)-yj
#         mp.AddQuadraticCost(dd.dot(dd))  #!!!!
#     result = Solve(mp)    #!!!!
#     print(result.is_success())    #!!!!
#     x=result.GetSolution(xi)  #!!!!
#     return x

def map(xi, cji, yj, reg):
    dd=cji.dot(xi)-yj
    return dd.dot(dd) + xi.dot(xi)*reg

def solve_one(m,cji_list,yj_list):
    # m, m.size, xi (>0, size=m.size)
    mp = MathematicalProgram()  #!!!!
    xi=mp.NewContinuousVariables(m.size, "xi")  #!!!!
    # print(m.size)
    # print(xi)
    for i in range(m.size):
        mp.AddLinearConstraint(xi[i] >= 0.)  #!!!!
    reg=0.1
    mp.AddQuadraticCost(xi.dot(xi)*reg)  #!!!!
    # print(cji_list)
    # print(yj_list)
    for i in range(len(cji_list)):
        cji=cji_list[i]
        yj=yj_list[i]
        dd=cji.dot(xi)-yj #+ 1e-2
        mp.AddQuadraticCost(dd.dot(dd))  #!!!! mp.AddQuadraticCost(dd.dot(dd))  #!!!!
        # print('cji: ', cji.shape)
        # print('yj: ', cji.shape)
        # print('dd.dot(dd): ', dd.dot(dd).shape)
    # print(mp)
    result = Solve(mp)    #!!!!
    # print(result.is_success())    #!!!!
    x=result.GetSolution(xi)  #!!!!
    # print(x)
    # print(x.shape)
    return x

# def mapping(fig_folder, fig_header, record_path, map_geometry, factor=factor1, save_process=True, savedata=True):
def mapping(fig_folder, fig_header, record_path, map_geometry, threshold, factor=factor1, save_process=True, savedata=True):
    recordpath = record_path+'_cal'   #!20220331
    map_horiz, map_vert  = map_geometry
    files=os.listdir(recordpath)
    files=sorted(files)
    # figurepath = 'mapping_data/save_fig/' + fig_header
    # figurepath_pdf = figurepath + '_pdf'
    # figurepath = f'{fig_folder}/{fig_header}'
    # figurepath_pdf = figurepath + '_pdf'
    if save_process:
        if not os.path.isdir(fig_folder):
            os.mkdir( fig_folder)
        # if not os.path.isdir(figurepath):
        #     os.mkdir( figurepath)
        # if not os.path.isdir(figurepath_pdf):
        #     os.mkdir(figurepath_pdf)
        os.system('rm -r ' +  fig_folder + "/*")    #!20220509
    # os.system('rm ' +  figurepath + "/*")    #!20220509
    # os.system('rm ' +  figurepath_pdf + "/*")    #!20220509
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
        yj_new = abs(yj)*factor    #!20220515

        cji_list.append(cji)
        yj_list.append(yj_new)

        RSID=data['RSID']
        # print('RSID', RSID)

        x=solve_one(m,cji_list,yj_list)
        print(filename)
        x_max = np.max(x)
        x=x/x_max
        # print(x)

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
            pos_x, pos_y, pos_dir, pos_ang = hxTrue_data[0,-1], hxTrue_data[1, -1], hxTrue_data[2, -1], hxTrue_data[3, -1]
            # arrow_x0 = 10*(hxTrue_data[0,-1] - hxTrue_data[0,-2])
            # arrow_y0 = 10*(hxTrue_data[1,-1] - hxTrue_data[1,-2])
            arrow_x0 = 1*np.cos(pos_dir)
            arrow_y0 = 1*np.sin(pos_dir)
            arrow_x1 = 1*np.cos(pos_ang)
            arrow_y1 = 1*np.sin(pos_ang)
            m.normalize()
            m.threshold(threshold)
            print(m.intensity)
            m.plot(fig_folder=fig_folder, fig_header=fig_header)
            for i in range(RSID.shape[0]):
                plt.plot(RSID[i, 0],RSID[i, 1],"xk",markersize=20)  #!20220804 multi sources
            # plt.plot(hxTrue_data[0,:], hxTrue_data[1, :], linewidth=2, color='#66CCCC')
            # plt.arrow(hxTrue_data[0,-2], hxTrue_data[1, -2], arrow_x, arrow_y, head_width = 0.8, width=0.1)
            # plt.arrow(pos_x, pos_y, arrow_x1, arrow_y1, head_width = 0.8, width=0.1, color='#D58B70')   # front side
            plt.arrow(pos_x, pos_y, arrow_x0, arrow_y0, head_width = 0.8, width=0.1, color='#77AE51')   # moving direction
            plt.arrow(pos_x, pos_y, arrow_x1, arrow_y1, head_width = 0.8, width=0.1, color='#8851AE')   # front side
            # plt.plot(hxTrue_data[0,-1], hxTrue_data[1, -1],"o", color='blue', markersize=7)
            # plt.plot(pos_x, pos_y,"o", color='blue', markersize=7)
            plt.plot(hxTrue_data[0,:], hxTrue_data[1, :], linewidth=2, color='#66CCCC') #  trajectory
            plt.plot(pos_x, pos_y,"o", color='#64ADB1', markersize=12)  # detector position
            plt.title('STEP: ' + filename[4:7], fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.tick_params(axis='both', which='minor', labelsize=15)
            plt.savefig(fname=fig_folder+'/'+filename[:7] +".png") #!20220323
            plt.savefig(fname=fig_folder+'/'+filename[:7] +".pdf") #!20220813
            plt.show()

    # m.intensity=x.reshape(m.x_num,m.y_num).T
    # m.plot()
    # for i in range(RSID.shape[0]):
    #     plt.plot(RSID[i, 0],RSID[i, 1],"xk")
    # plt.savefig(fname=fig_folder+'/'+fig_header+"_plot2.png")
    # plt.show()
    # print("Plot2")

    # with imageio.get_writer(fig_folder+'/'+fig_header+'.gif', mode='I') as writer:
    #     for figurename in sorted(os.listdir(figurepath)):
    #         image = imageio.imread(figurepath + '/' + figurename)
    #         writer.append_data(image)
    # pass

# def test():
#     m=Map(map_horiz, map_vert)
#     m.intensity=np.random.rand(m.x_num,m.y_num).T
#     m.plot()
#     plt.show()
#     plt.savefig(fname=fig_folder+'/'+fig_header+"_plot3.png")
#     plt.savefig(fname=fig_folder+'/'+fig_header+"_plot3.pdf")

def gen_gif(fig_folder):
    # figurepath = 'mapping_data/save_fig/' + fig_header
    with imageio.get_writer(f'{fig_folder}/mapping.gif', mode='I') as writer:
        for figurename in sorted(os.listdir(fig_folder)):
            if figurename.endswith('png'):
                image = imageio.imread(fig_folder + '/' + figurename)
                writer.append_data(image)
    print( f'Finish making a gif: {fig_folder}/mapping.gif')

    
#%%
# if __name__ == '__main__':
#     main()
#     # test()
#     gen_gif()
        
# %%
