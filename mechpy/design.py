# -*- coding: utf-8 -*-
'''
scripts and boilerplate code to use for mechanical engineering design tasks
'''

__author__ = 'Neal Gordon <nealagordon@gmail.com>'
__date__ =   '2016-09-06'

import math
import pandas as pd
import numpy as np
from numpy import pi, array
from numpy.linalg import eig
import matplotlib.pyplot as plt

def gear():
    '''
    Plotting of the Involute function applied to gear
    design (adapated from http://www.arc.id.au/GearDrawing.html)    
    Transcendental Parametric function describing the contour of the gear face
    '''    

    th1 = np.pi/4
    th2 = np.pi/3
    thcir = np.linspace(-np.pi,np.pi,100)
    th = np.linspace(th1,th2,100)
    Rb = 0.5
    x = Rb*(np.sin(th)+np.cos(th))
    y = Rb*(np.sin(th)-np.cos(th))
    xcir = np.sin(thcir)
    ycir = np.cos(thcir)
    
    ofst = 0.05
    y = max(ycir)+y
    x = x-min(x)+ofst
    
    plt.plot(x,y)
    plt.plot(-x,y)
    plt.plot([-ofst , ofst],[max(y) , max(y)] )
    
    plt.plot(xcir,ycir,'--')
    plt.show()


def fastened_joint(fx, fy, P, l):
    '''computes stressed in fastened joints with bolts or rivets
    INCOMPLETE
    
    # fastener location
    fx = array([0,1,2,3,0,1,2,3])
    fy = array([0,0,0,0,1,1,1,1])
    # Force(x,y)
    P = array([-300,-500])
    l = [2,1]
    
    '''
    
    fn = range(len(fx))
    
    df = pd.DataFrame()
    
    Pnorm = P/np.max(np.abs(P))  # for plotting
    # Location of Force P, x,y
    
    d = array(5/16*np.ones(len(fn)))
    
    A = array([pi*d1**2/4 for d1 in d])
    
    fn = range(len(fx))
    
    df = pd.DataFrame({ 'Fastener' : fn,
                         'x' : fx,
                         'y' : fy}, index=fn)
                  
    df['x^2'] = df.x**2
    df['y^2'] = df.y**2
    df['xbar'] = np.sum(A*fx)/np.sum(A)
    df['ybar'] = np.sum(A*fy)/np.sum(A)    
    return df


def mohr(sx, sy, txy):
    cen = (sx + sy)*.5
    rad = math.sqrt(((sx - sy)*.5)**2 +  txy ** 2)
    s1 = round(cen + rad, ndigits=2)
    s2 = round(cen - rad, ndigits=2)

    # Plotting
    off = .20*rad
    xaxis = np.linspace(cen-rad-off, cen+rad+off)
    yaxis = np.linspace(-(off+rad), +off+rad)
    plt.plot(xaxis, 0*xaxis)
    plt.plot(cen+0*yaxis, yaxis)
    t = np.arange(0, np.pi * 2.0, 0.01)
    x = cen+rad * np.cos(t)
    y = rad * np.sin(t)
    #img = plt.imread("mohr.png")    # TODO improve bg image
    #plt.imshow(img, extent=[(cen-rad), cen+rad, -rad, rad])
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_aspect('equal')  # need circle, not a ellipse
    plt.annotate('$\sigma_1=$'+str(s1), xy=(cen+rad, 0), xytext=(cen+rad/2, rad/2),
                   arrowprops=dict ( facecolor='black', shrink=0.01 )
                  )
    plt.annotate('$\sigma_2=$'+ str(s2), xy=(cen-rad, 0), xytext=(cen - rad/2, rad / 2),
                   arrowprops=dict ( facecolor='red', shrink=0.01 )
                  )
    #TODO annotate max shear
    plt.show()

def mohr3D(sxx,syy,szz,sxy,sxz,syz):
    stress = np.array([[sxx,sxy,sxz],
                       [sxy,syy,syz],
                       [sxz,syz,szz]])
    eign  = eig(stress)
    pstress =eign[0]
    pstress.sort()

    # define circles
    circ = np.zeros ( (3, 2), dtype=float )  # 3 circles in order center,radius
    circ[ 0 ][ 0 ] = .5 * (pstress[ 2 ] + pstress[ 0 ])
    circ[ 0 ][ 1 ] = .5 * (pstress[ 2 ] - pstress[ 0 ])
    circ[ 1 ][ 0 ] = .5 * (pstress[ 1 ] + pstress[ 0 ])
    circ[ 1 ][ 1 ] = .5 * (pstress[ 1 ] - pstress[ 0 ])
    circ[ 2 ][ 0 ] = .5 * (pstress[ 2 ] + pstress[ 1 ])
    circ[ 2 ][ 1 ] = .5 * (pstress[ 2 ] - pstress[ 1 ])

    # Plotting
    circle1 = plt.Circle((circ[0][0],0), radius = circ[0][1], color ='red')
    circle2 = plt.Circle((circ[1][0],0), radius = circ[1][1], color ='blue')
    circle3 = plt.Circle((circ[2][0],0), radius = circ[2][1], color ='green')
    ax = plt.gca()
    ax.add_patch(circle1)
    ax.add_patch ( circle2 )
    ax.add_patch ( circle3 )
    # TODO Add axes and annotate
    plt.axis('scaled')
    plt.show()


if __name__=='__main__':
    #shear_bending()
    #mohr(10, 40, 15)
    mohr3D(10, -10, 10, -5, 5, -5)
