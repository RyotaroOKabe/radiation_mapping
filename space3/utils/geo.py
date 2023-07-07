import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as matplotlib_polygon
import matplotlib
import math

#from crossing import *
from utils.crossing import *



class Point(tuple):
    """docstring for Point"""
    def __new__ (self, x, y):
        return super(Point, self).__new__(self, tuple([float(x),float(y)]))

    def __init__(self, x,y):
        self.x = float(x)
        self.y = float(y)

    def plot(self,ax=None,**args):
        if not ax:ax=plt.gca()
        ax.plot([self.x],[self.y],**args)
    
class Edge(tuple):
    """docstring for Edge"""
    def __new__ (self, p1, p2):
        return super(Edge, self).__new__(self, tuple([p1,p2]))

    def __init__(self, p1,p2):
        self.x1 = p1.x
        self.y1 = p1.y
        self.x2 = p2.x
        self.y2 = p2.y
        self.p1 = p1
        self.p2 = p2
    def plot(self,ax=None,**args):
        if not ax:ax=plt.gca()
        ax.plot([self.x1,self.x2],[self.y1,self.y2],**args)

class Ray(tuple):
    """docstring for Edge"""
    def __new__ (self, p1, arg):
        p2=Point(p1[0]+10.0*math.cos(arg),p1[1]+10.0*math.sin(arg))
        return super(Ray, self).__new__(self, tuple([p1,p2]))

    def __init__(self, p1 , arg):
        self.arg=arg
        self.x1 = p1.x
        self.y1 = p1.y      
        self.p1 = p1
        self.p2 = Point(p1[0]+10.0*math.cos(arg),p1[1]+10.0*math.sin(arg))
        self.x2 = self.p2.x
        self.y2 = self.p2.y

    def plot(self,length,ax=None,**args):
        if not ax:ax=plt.gca()
        x2=self.x1+length*math.cos(self.arg)
        y2=self.y1+length*math.sin(self.arg)
        ax.plot([self.x1,x2],[self.y1,y2],**args)
        
class Polygon(object):
    """docstring for Polygon"""
    def __init__(self, apex):
        super(Polygon, self).__init__()
        self.apex = apex
        self.edge = []
        for i in range(len(apex)-1):
            self.edge.append(Edge(apex[i],apex[i+1]))
        self.edge.append(Edge(apex[-1],apex[0]))
    
    def contain_point(self,point):
        pass
    def check_convex(self):
        pass
    
    def plot(self,rotation=None,ax=None,**args):
        if not ax:ax=plt.gca()

        apexes=[]
        if rotation: 
            for apex in self.apex:
                x=np.cos(rotation)*apex[0]-np.sin(rotation)*apex[1]
                y=np.sin(rotation)*apex[0]+np.cos(rotation)*apex[1]
                apexes.append([x,y])
        else:
            apexes=self.apex

        self.patch=matplotlib_polygon(apexes,closed=True,fill=True,**args)
        ax.add_patch(self.patch)
        ax.autoscale_view()

def get_crossing(line1,line2):
    a1,b1,c1 = getLinePara(line1)
    a2,b2,c2 = getLinePara(line2)
    d = a1* b2 - a2 * b1
    p = [0,0]
    if d == 0:
        if b1*c2-b2*c1==0:
            return ()
        else: return None
    else:
        p[0] = (b1 * c2 - b2 * c1)*1.0 / d
        p[1] = (c1 * a2 - c2 * a1)*1.0 / d
    p = Point(p[0],p[1])
    #print p
    ff=inSegment
    l1=line1;l2=line2
    if type(line1)==Ray:
        ff=inSegment_ray
        
    elif type(line2)==Ray:
        ff=inSegment_ray
        l1=line2;l2=line1
    if ff(p,line1,line2):
        return p
    else:
        if getLineLength(p,l1[0]) ==0 or getLineLength(p,l1[1]) ==0:
            pass

        return None

def cross_line_polygon2(line,poly):
    ps=[]
    for line2 in poly.edge:
        p=get_crossing(line,line2)
        if p:
            ps.append(p)
    ps=sorted(ps,key=lambda p:get_len(line[0],p))
    return ps
    pass

def cross_line_polygon(line,poly):
    ps=[]
    for line2 in poly.edge:
        p=get_crossing(line,line2)
        if p:ps.append(get_len(line[0],p))
    ps=sorted(ps)
    return ps
    pass

def get_arg(*args):
    if len(args)==1:
        p1=args[0][0]
        p2=args[0][1]
    elif len(args)==2:
        p1=args[0]
        p2=args[1]
    else:
        return None
    if p1[0]==p2[0] and p1[1]==p2[1]:
        return None
    if p1[0]==p2[0]:
        if p2[1]>p1[1]:
            return math.pi/2
        else:
            return -math.pi/2
    return math.atan((p2[1]-p1[1])/(p2[0]-p1[0]))

def get_len(*args):
    if len(args)==1:
        p1=args[0][0]
        p2=args[0][1]
    elif len(args)==2:
        p1=args[0]
        p2=args[1]
    else:
        return None
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

if __name__ == '__main_':
    a=Point(1.99,2.29)
    b=Point(3.4,5.56)
    c=Point(2.3,5)
    d=Point(3,-2)

    e=Point(1.,1.)
    f=Point(3.2,6.3)

    cmap = matplotlib.cm.get_cmap('hot')
    sb=Ray(Point(-1,-0.45),0.1)
    sb=Edge(Point(-1,-0.45),Point(-1+2.0*math.cos(0.1),-0.45+2.*math.sin(0.1)))
    sb.plot()
    jb=Polygon([a,d,b,c])
    jb=Polygon([Point(-0.2955555555555555, -0.3977777777777778), Point(-0.2955555555555555, 0.3177777777777778), Point(-0.21555555555555553, -0.3177777777777778), Point(-0.21555555555555553, -0.3977777777777778)])
    jb.plot(facecolor=cmap(0.6))
    print(cross_line_polygon2(sb,jb))
    print(cross_line_polygon(sb,jb))

    l1=Ray(a,3.14/3)
    l2=Edge(c,d)
    plt.show()

if __name__ == '__main__':
    poly=Polygon([Point(3.0, -2.0), Point(2.0, -2.0), Point(2.0, -3.0), Point(3.0, -3.0)])
    line2=Edge(Point(2.0, -2.0), Point(2.0, -3.0))
    line2.plot()
    r=Ray(Point(2.0, 2.0), -1.25663706144)
    r.plot(4)
    print(get_crossing(line2,r))
    print(inSegment_ray(Point(2.0, 2.0),r,line2))
    plt.show()