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

# minazi=270
# maxazi=275
# minazi=155
# maxazi=160
# mindist=1075
# maxdist=1125
# fname='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_TravelTime_MAP/Amp_Dist_NKNT.'\
#     +str(mindist)+'_'+str(maxdist)+'.txt';
# degree_sign= u'\N{DEGREE SIGN}'
# PlotFile(fname, FMT='o', x=3, y1=5, y2=None, err1=None, err2=None, lw=10, markersize=10)
# # plt.title('Amplitude for 10 sec: 1075 km < dist < 1125 km')
# # plt.ylabel('Amplitude(nm)',fontsize=20)
# # plt.title('Amplitude for 10 sec ('+str(mindist)+' km < D <'+str(maxdist)+' km)',fontsize=35)
# plt.ylabel('Ms',fontsize=20)
# plt.title('Surface Wave Magnitude for 10 sec ('+str(mindist)+' km < D <'+str(maxdist)+' km)',fontsize=28)
# plt.xlabel('Azimuth('+degree_sign+')', fontsize=20)
# 
# # plt.title('Surface Wave Magnitude for 10 sec:' +str(minazi)+' < az <'+str(maxazi))
# plt.show()

prefix='/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_TravelTime_MAP/Amp_Dist_NKNT.'
degree_sign= u'\N{DEGREE SIGN}'
# fname='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_TravelTime_MAP/Amp_Azi_NKNT.1075_1125.txt';
# minazi=270
# maxazi=275
# minazi=155
# maxazi=160
minazi=250
maxazi=255
fname=prefix+str(minazi)+'_'+str(maxazi)+'.txt';
ax=plt.subplot(111)
line1=PlotFile(fname, ax=ax, FMT='ro', x=3, y1=5, y2=None, err1=None, err2=None, lw=10, markersize=25)
minazi=270
maxazi=275
fname=prefix+str(minazi)+'_'+str(maxazi)+'.txt';
line2=PlotFile(fname, ax=ax, FMT='bo', x=3, y1=5, y2=None, err1=None, err2=None, lw=10, markersize=25)
minazi=155
maxazi=160
fname=prefix+str(minazi)+'_'+str(maxazi)+'.txt';
line3=PlotFile(fname, ax=ax, FMT='yo', x=3, y1=5, y2=None, err1=None, err2=None, lw=10, markersize=25)
ax.legend([line1, line2, line3], ['250'+degree_sign+'~255'+degree_sign,\
        '270'+degree_sign+'~275'+degree_sign, \
        '155'+degree_sign+'~165'+degree_sign], loc=0)
# plt.title('Amplitude for 10 sec: 1075 km < dist < 1125 km')
# plt.ylabel('Amplitude(nm)',fontsize=20)
# plt.title('Amplitude for 10 sec', fontsize=28)
plt.ylabel('Ms',fontsize=20)
plt.title('Surface Wave Magnitude for 10 sec',fontsize=28)
plt.xlabel('Distance(km)', fontsize=20)
plt.xlim(400, 1400)
# plt.ylim(100, 400)
plt.ylim(3.4, 4.1)

# plt.title('Surface Wave Magnitude for 10 sec:' +str(minazi)+' < az <'+str(maxazi))
plt.show()


