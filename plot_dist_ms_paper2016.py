# -*- coding: utf-8 -*-
"""
Quick and dirty python plot script
By Lili Feng
"""
import numpy as np
import matplotlib.pyplot as plt
import os;
# from mpl_toolkits.basemap import Basemap


# def ls(indir='.'):
#     os.listdir(indir)
#     return

def PlotFile(fname, ax=plt.subplot(), FMT='', x=1, y1=2, y2=None, err1=None, err2=None, lw=3, markersize=5):
    Inarray=np.loadtxt(fname);
    X=Inarray[:,x-1];
    Y1=Inarray[:,y1-1];
    # ax=plt.subplot()
    if err1==None:
        if FMT!='':
            line,=ax.plot(X, Y1, FMT, linewidth=lw, markersize=markersize);
        else:
            ax.plot(X, Y1, linewidth=lw, markersize=markersize);
    else:
        errin1=Inarray[:,err1-1]
        if FMT!='':
            plt.errorbar(X, Y1, fmt=FMT, linewidth=lw, yerr=errin1)
        else:
            plt.errorbar(X, Y1, fmt='.', linewidth=lw, yerr=errin1)
            
    if y2!=None:
        Y2=Inarray[:,y2-1];
        if err2==None:
            if FMT!='':
                ax.plot(X, Y2, fmt=FMT, linewidth=lw);
            else:
                ax.plot(X, Y2, linewidth=lw);
        else:
            errin2=Inarray[:,err2-1]
            if FMT!='':
                plt.errorbar(X, Y2, fmt=FMT, linewidth=lw, yerr=errin2)
            else:
                plt.errorbar(X, Y2, fmt='.', linewidth=lw, yerr=errin2)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    return line

def CompareFiles(fname1, fname2, FMT='', DIF=False, x1=1, y1=2, x2=1, y2=2, err1=None, err2=None, lw=3):
    Inarray1=np.loadtxt(fname1);
    X1=Inarray1[:,x1-1];
    Y1=Inarray1[:,y1-1];
    Inarray2=np.loadtxt(fname2);
    X2=Inarray2[:,x2-1];
    Y2=Inarray2[:,y2-1];
    ax=plt.subplot()
    if err1==None:
        if FMT!='':
            line, = ax.plot(X1, Y1, fmt=FMT, linewidth=lw);
        else:
            ax.plot(X1, Y1,'--r', linewidth=lw);
    else:
        errin1=Inarray1[:,err1-1]
        if FMT!='':
            plt.errorbar(X1, Y1, fmt=FMT, linewidth=lw, yerr=errin1)
        else:
            plt.errorbar(X1, Y1, fmt='.', linewidth=lw, yerr=errin1)
            
    if err2==None:
        if FMT!='':
            ax.plot(X2, Y2, fmt=FMT,linewidth=lw);
        else:
            ax.plot(X2, Y2,'-.b', linewidth=lw);
    else:
        errin2=Inarray[:,err2-1]
        if FMT!='':
            plt.errorbar(X2, Y2, fmt=FMT, linewidth=lw, yerr=errin2)
        else:
            plt.errorbar(X2, Y2, fmt='.', linewidth=lw, yerr=errin2)
    
    if DIF==True:
        plt.figure()
        ax2=plt.subplot()
        difarray=Y1-Y2;
        if FMT!='':
            ax2.plot(X1, difarray, fmt=FMT, linewidth=lw);
        else:
            ax2.plot(X1, difarray, 'x', linewidth=lw);
    return line
            
    
def Plot3DFile(fname, CMAP='gist_rainbow', x=1, y=2, z=3):
    Inarray=np.loadtxt(fname);
    X=Inarray[:,x-1];
    Y=Inarray[:,y-1];
    Z=Inarray[:,z-1];
    ax=plt.subplot()
    p=plt.tripcolor(X, Y, Z, cmap=CMAP)
    plt.colorbar(p, ax=ax)
    return

def Compare3DFiles(fname1, fname2, CMAP='gist_rainbow', x=1, y=2, z1=3, z2=3):
    Inarray=np.loadtxt(fname1);
    X=Inarray[:,x-1];
    Y=Inarray[:,y-1];
    Z1=Inarray[:,z1-1];
    Inarray2=np.loadtxt(fname2);
    Z2=Inarray2[:,z2-1];
    difZ=Z1-Z2
    ax=plt.subplot()
    p=plt.tripcolor(X, Y, difZ, cmap=CMAP)
    plt.colorbar(p, ax=ax)
    return




def Show():
    plt.show()
markersize=25
plt.figure(figsize=(20,10))
datadir='/projects/life9360/code/SES3DPy/ms_experiment'
degree_sign= u'\N{DEGREE SIGN}'
mindist=475.
maxdist=525.
fname=datadir+'/dist_'+str(mindist)+'_'+str(maxdist)+'_amp_Ms.lst'
ax=plt.subplot(111)
line1=PlotFile(fname, ax=ax, FMT='bo', x=1, y1=3, y2=None, err1=None, err2=None, lw=10, markersize=markersize)
mindist=1075.
maxdist=1125.
fname=datadir+'/dist_'+str(mindist)+'_'+str(maxdist)+'_amp_Ms.lst'
line2=PlotFile(fname, ax=ax, FMT='go', x=1, y1=3, y2=None, err1=None, err2=None, lw=10, markersize=markersize)
mindist=1675.
maxdist=1725.
fname=datadir+'/dist_'+str(mindist)+'_'+str(maxdist)+'_amp_Ms.lst'
line3=PlotFile(fname, ax=ax, FMT='ro', x=1, y1=3, y2=None, err1=None, err2=None, lw=10, markersize=markersize)


ax.legend([line1, line2, line3], ['475 km'+'~ 525 km',\
        '1075 km'+'~ 1125 km', \
        '1675 km'+'~ 1725 km'], loc=1, numpoints = 1, fontsize=30)
plt.ylabel('Ms',fontsize=30)
plt.title('Surface Wave Magnitude: T = 10 sec',fontsize=38)
plt.xlabel('Azimuth('+degree_sign+')', fontsize=30)
plt.ylim(2.8, 3.7)
plt.yticks(np.arange(2.8, 3.7, 0.1))
plt.xlim(140., 360.)
plt.xticks(np.arange(140, 360+1, 20.0))

# plt.title('Surface Wave Magnitude for 10 sec:' +str(minazi)+' < az <'+str(maxazi))
plt.show()#!/usr/bin/env python

