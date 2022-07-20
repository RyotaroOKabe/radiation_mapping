#%%
# -*- coding: UTF-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import sys,os
import pickle as pkl

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
def test():
    plt.figure()

    cmap = matplotlib.cm.get_cmap('gray')

    m=Map([-5,5,10],[-5,5,10])
    pose=np.array([2.2,2.2,np.pi/2]).reshape((3,1))

    cji=cal_cji(m,pose)

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

def write_data(recordpath): #!20220717
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

        cji=cal_cji(m,pose)
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
    #test()
    shape_name = 'J'
    recordpath = f'mapping_data/mapping_A20220720_{shape_name}_v2.1'   #!20220331
    write_data(recordpath)  #!20220717
    # ss=Point(-2,0)
    # r1=Ray(ss,-np.pi*0-0.05)
    # r2=Ray(ss,-np.pi*0+0.06)
    # r1.plot(4)
    # r2.plot(4)
    # sq=Square([0,0],[1,5])
    # sq.plot(color='yellow')
    # print sq.apex
    # print square_intersect(sq,r1,r2)
    #plt.show()
    