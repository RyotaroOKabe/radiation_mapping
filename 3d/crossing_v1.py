#coding=utf-8
from math import sqrt

float_err=1e-10

def inSegment(p,line,line2):

    if line[0][0] == line[1][0]:
        if  p[1] > min(line[0][1],line[1][1]) and p[1] < max(line[0][1],line[1][1]):
            if p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
                return True
    elif line[0][1] == line[1][1]:
        if p[0] > min(line[0][0],line[1][0]) and p[0] < max(line[0][0],line[1][0]):

            if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]):
                return True
    else:
        if p[0] > min(line[0][0],line[1][0]) and p[0] < max(line[0][0],line[1][0]):

            if line2[0][0] == line2[1][0]:
                if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]):
                    return True
            elif line2[0][1] == line2[1][1]:
                if p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
                    return True
            else:
                if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]) and p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
                    return True
    return False

def inSegment_ray(p,line,line2):
    '''
    line is ray, 0th element is source
    '''
    if line[0][0] == line[1][0]:
        if  (p[1] - line[0][1])*(line[1][1]-line[0][1]) >= 0:
            if p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
                return True
    elif line[0][1] == line[1][1]:
        if  (p[0] - line[0][0])*(line[1][0]-line[0][0]) >= 0:
            if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]):
                return True
    else:
        if  (p[0] - line[0][0])*(line[1][0]-line[0][0]) >= 0:
            if line2[0][0] == line2[1][0]:
                if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]):
                    return True
            elif line2[0][1] == line2[1][1]:
                if p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
                    return True
            else:
                if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]) and p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
                    return True
    return False

def getLinePara(line):
    a = line[0][1] - line[1][1]
    b = line[1][0] - line[0][0]
    c = line[0][0] *line[1][1] - line[1][0] * line[0][1]
    return a,b,c

def getLineLength(p1,p2):
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def getCrossPoint(line1,line2):
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
    p = tuple(p)
    if inSegment(p,line1,line2) or getLineLength(p,line1[0]) ==0 or getLineLength(p,line1[1]) ==0:
        return p
    else:
        return None

if __name__ == '__main__':

    print(getCrossPoint([(1.0,-3.0),(1.0,5.0)],[(0.,2),(2.0,2.0)]))
    print(getCrossPoint([(1.0,-3.0),(1.0,5.0)],[(1,-6),(1,7.0)]))
    print(inSegment_ray((-5.,0.),[(-2.,0.),(-1.,0.)],[(-5.,-1),(-5.,1)]))
