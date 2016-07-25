
"""
A python module serve as input preparation and post-precessing code for SES3D and AxiSEM

:Methods:
    
:Dependencies:
    numpy 1.9.1
    matplotlib 1.4.3
    numexpr 2.3.1
    ObsPy 0.10.2
    pyfftw3 0.2.1
    pyaftan( compiled from aftan 1.1 )
    GMT 5.x.x (For Eikonal/Helmholtz Tomography)
    
    
:Modified and restructrued from original python script in ses3d package( by Andreas Fichtner & Lion Krischer )
    several new functions added.
    
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

from wfs_input_generator import InputFileGenerator
import obspy.signal.filter as flt
import obspy
import numpy as np
import obspy.core.util.geodetics as obsGeo
import matplotlib.pyplot as plt
import fftw3 # pyfftw3-0.2
# from matplotlib.mlab import griddata
from scipy.interpolate import griddata
# from netCDF4 import Dataset
import obspy.sac.sacio as sacio
import obspy.taup.taup
import fftw3 # pyfftw3-0.2
import numexpr as npr
import math
import scipy.signal
import numpy.random as rd
import mpl_toolkits.basemap 
from mpl_toolkits.basemap import Basemap, shiftgrid
from pylab import *
import glob, os
from matplotlib.pyplot import cm
import matplotlib.pylab as plb
import copy
from scipy import interpolate

from functools import partial
import multiprocessing as mp

import time
import shutil
from subprocess import call

import warnings

###################################################################################################
#- rotation matrix
###################################################################################################
def rotation_matrix(n,phi):

    """ compute rotation matrix
    input: rotation angle phi [deg] and rotation vector n normalised to 1
    return: rotation matrix
    """
    phi=np.pi*phi/180.0
    A=np.array([ (n[0]*n[0],n[0]*n[1],n[0]*n[2]), (n[1]*n[0],n[1]*n[1],n[1]*n[2]), (n[2]*n[0],n[2]*n[1],n[2]*n[2])])
    B=np.eye(3)
    C=np.array([ (0.0,-n[2],n[1]), (n[2],0.0,-n[0]), (-n[1],n[0],0.0)])
    R=(1.0-np.cos(phi))*A+np.cos(phi)*B+np.sin(phi)*C
    return np.matrix(R)
###################################################################################################
#- rotate coordinates
###################################################################################################
def rotate_coordinates(n,phi,colat,lon):
    """ rotate colat and lon
    input: rotation angle phi [deg] and rotation vector n normalised to 1, original colatitude and longitude [deg]
    return: colat_new [deg], lon_new [deg]
    """
    # convert to radians
    colat=np.pi*colat/180.0
    lon=np.pi*lon/180.0
    # rotation matrix
    R=rotation_matrix(n,phi)
    # original position vector
    x=np.matrix([[np.cos(lon)*np.sin(colat)], [np.sin(lon)*np.sin(colat)], [np.cos(colat)]])
    # rotated position vector
    y=R*x
    # compute rotated colatitude and longitude
    colat_new=np.arccos(y[2])
    lon_new=np.arctan2(y[1],y[0])
    return float(180.0*colat_new/np.pi), float(180.0*lon_new/np.pi)
###################################################################################################
#- rotate moment tensor
###################################################################################################
def rotate_moment_tensor(n,phi,colat,lon,M):
    """ rotate moment tensor
    input: rotation angle phi [deg] and rotation vector n normalised to 1, original colat and lon [deg], original moment tensor M as matrix
    M=[Mtt Mtp Mtr
       Mtp Mpp Mpr
       Mtr Mpr Mrr]
    return: rotated moment tensor
    """
    # rotation matrix
    R=rotation_matrix(n,phi)
    # rotated coordinates
    colat_new,lon_new=rotate_coordinates(n,phi,colat,lon)
    # transform to radians
    colat=np.pi*colat/180.0
    lon=np.pi*lon/180.0
    colat_new=np.pi*colat_new/180.0
    lon_new=np.pi*lon_new/180.0
    # original basis vectors with respect to unit vectors [100].[010],[001]
    bt=np.matrix([[np.cos(lon)*np.cos(colat)],[np.sin(lon)*np.cos(colat)],[-np.sin(colat)]])
    bp=np.matrix([[-np.sin(lon)],[np.cos(lon)],[0.0]])
    br=np.matrix([[np.cos(lon)*np.sin(colat)],[np.sin(lon)*np.sin(colat)],[np.cos(colat)]])
    # original basis vectors with respect to rotated unit vectors
    bt=R*bt
    bp=R*bp
    br=R*br
    # new basis vectors with respect to rotated unit vectors
    bt_new=np.matrix([[np.cos(lon_new)*np.cos(colat_new)],[np.sin(lon_new)*np.cos(colat_new)],[-np.sin(colat_new)]])
    bp_new=np.matrix([[-np.sin(lon_new)],[np.cos(lon_new)],[0.0]])
    br_new=np.matrix([[np.cos(lon_new)*np.sin(colat_new)],[np.sin(lon_new)*np.sin(colat_new)],[np.cos(colat_new)]])
    # assemble transformation matrix and return
    A=np.matrix([[float(bt_new.transpose()*bt), float(bt_new.transpose()*bp), float(bt_new.transpose()*br)],\
          [float(bp_new.transpose()*bt), float(bp_new.transpose()*bp), float(bp_new.transpose()*br)],\
          [float(br_new.transpose()*bt), float(br_new.transpose()*bp), float(br_new.transpose()*br)]])
    return A*M*A.transpose()




"""
colormap for creating custom color maps.  For example...
  >>> from pyclaw.plotting import colormaps
  >>> mycmap = colormaps.make_colormap({0:'r', 1.:'b'})  # red to blue
  >>> colormaps.showcolors(mycmap)   # displays resulting colormap

Note that many colormaps are also defined in matplotlib and can be set by
  >>> from matplotlib import cm
  >>> mycmap = cm.get_cmap('Greens')
for example, to get colors ranging from white to green.
See matplotlib._cm for the data defining various maps.
"""
#-------------------------
def make_colormap(colors):
#-------------------------
    """
    Define a new color map based on values specified in the dictionary
    colors, where colors[z] is the color that value z should be mapped to,
    with linear interpolation between the given values of z.
    The z values (dictionary keys) are real numbers and the values
    colors[z] can be either an RGB list, e.g. [1,0,0] for red, or an
    html hex string, e.g. "#ff0000" for red.
    """
    from matplotlib.colors import LinearSegmentedColormap, ColorConverter
    from numpy import sort
    z = sort(colors.keys())
    n = len(z)
    z1 = min(z)
    zn = max(z)
    x0 = (z - z1) / (zn - z1)
    CC = ColorConverter()
    R = []
    G = []
    B = []
    for i in range(n):
        #i'th color at level z[i]:
        Ci = colors[z[i]]
        if type(Ci) == str:
            # a hex string of form '#ff0000' for example (for red)
            RGB = CC.to_rgb(Ci)
        else:
            # assume it's an RGB triple already:
            RGB = Ci
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])
    cmap_dict = {}
    cmap_dict['red'] = [(x0[i],R[i],R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(x0[i],G[i],G[i]) for i in range(len(G))]
    cmap_dict['blue'] = [(x0[i],B[i],B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap',cmap_dict)
    return mymap

def showcolors(cmap):
    from pylab import colorbar, clf, axes, linspace, pcolor, \
         meshgrid, show, axis, title
    #from scitools.easyviz.matplotlib_ import colorbar, clf, axes, linspace,\
                 #pcolor, meshgrid, show, colormap
    clf()
    x = linspace(0,1,21)
    X,Y = meshgrid(x,x)
    pcolor(X,Y,0.5*(X+Y), cmap=cmap, edgecolors='k')
    axis('equal')
    colorbar()
    title('Plot of x+y using colormap')
    return;

def schlieren_colormap(color=[0,0,0]):
    """
    For Schlieren plots:
    """
    from numpy import linspace, array
    if color=='k': color = [0,0,0]
    if color=='r': color = [1,0,0]
    if color=='b': color = [0,0,1]
    if color=='g': color = [0,0.5,0]
    color = array([1,1,1]) - array(color)
    s  = linspace(0,1,20)
    colors = {}
    for key in s:
        colors[key] = array([1,1,1]) - key**10 * color
    schlieren_colors = make_colormap(colors)
    return schlieren_colors

# -----------------------------------------------------------------
# Some useful colormaps follow...
# There are also many colormaps in matplotlib.cm
all_white = make_colormap({0.:'w', 1.:'w'})
all_light_red = make_colormap({0.:'#ffdddd', 1.:'#ffdddd'})
all_light_blue = make_colormap({0.:'#ddddff', 1.:'#ddddff'})
all_light_green = make_colormap({0.:'#ddffdd', 1.:'#ddffdd'})
all_light_yellow = make_colormap({0.:'#ffffdd', 1.:'#ffffdd'})

red_white_blue = make_colormap({0.:'r', 0.5:'w', 1.:'b'})
blue_white_red = make_colormap({0.:'b', 0.5:'w', 1.:'r'})
red_yellow_blue = make_colormap({0.:'r', 0.5:'#ffff00', 1.:'b'})
blue_yellow_red = make_colormap({0.:'b', 0.5:'#ffff00', 1.:'r'})
yellow_red_blue = make_colormap({0.:'#ffff00', 0.5:'r', 1.:'b'})
white_red = make_colormap({0.:'w', 1.:'r'})
white_blue = make_colormap({0.:'w', 1.:'b'})

schlieren_grays = schlieren_colormap('k')
schlieren_reds = schlieren_colormap('r')
schlieren_blues = schlieren_colormap('b')
schlieren_greens = schlieren_colormap('g')


class ses3dStream(obspy.core.stream.Stream):
    
    def Getses3dsynDIST(self, datadir, SLst, mindist, maxdist, channel='BXZ'):
        for station in SLst.stations:
            # LF.EA116S42..BXZ.SAC
            sacfname=datadir+'/'+station.network+'.'+station.stacode+'..'+channel+'.SAC';
            if not os.path.isfile(sacfname):
                # print 'Not: '+sacfname;
                continue;
            trace=obspy.read(sacfname)[0];
            
            if trace.stats.sac.dist<mindist or trace.stats.sac.dist>maxdist:
                continue;
            print 'Reading: '+station.network+'.'+station.stacode+'..'+channel+'.SAC';
            self.traces.append(trace);
        return;
    
    
    def PlotDISTStreams(self, ampfactor=0.01, title='', ax=plt.subplot(), targetDT=0.1):
        ymax=-999.;
        ymin=999.;
        
        for trace in self.traces:
            downsamplefactor=int(targetDT/trace.stats.delta)
            # trace.decimate(factor=downsamplefactor, no_filter=True);
            dt=trace.stats.delta;
            time=dt*np.arange(trace.stats.npts);
            
            yvalue=trace.data*ampfactor;
            azi=float(trace.stats.sac.az);
            ax.plot(time, yvalue+azi, '-k', lw=1);
            tfill=time[yvalue>0];
            yfill=(yvalue+azi)[yvalue>0];
            ax.fill_between(tfill, azi, yfill, color='blue', linestyle='--', lw=0.);
            tfill=time[yvalue<0];
            yfill=(yvalue+azi)[yvalue<0];
            ax.fill_between(tfill, azi, yfill, color='red', linestyle='--', lw=0.);
            pos='lon: '+str(trace.stats.sac.stlo)+' lat: '+str(trace.stats.sac.stla)
            # plt.text(5, azi, pos)
            ymin=min(ymin, azi)
            ymax=max(ymax, azi)
            Valuemax=(npr.evaluate('abs(yvalue/ampfactor)')).max()
            # plt.text(time.max()-100, azi, str(Valuemax)+' nm')
        ymin=ymin;
        ymax=ymax;
        plt.axis([0., time.max(), ymin, ymax])
        plt.xlabel('Time(sec)');
        plt.title(title);
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        # fig.savefig(outdir+'/ses3d_ST.pdf', orientation='landscape', format='pdf')
        return;
    
    def Getses3dsynAzi(self, datadir, SLst, minazi, maxazi, channel='BXZ'):
        for station in SLst.stations:
            # LF.EA116S42..BXZ.SAC
            sacfname=datadir+'/'+station.network+'.'+station.stacode+'..'+channel+'.SAC';
            if not os.path.isfile(sacfname):
                # print 'Not: '+sacfname;
                continue;
            trace=obspy.read(sacfname)[0];
            if trace.stats.sac.az<minazi or trace.stats.sac.az>maxazi:
                continue;
            print 'Reading: '+station.network+'.'+station.stacode+'..'+channel+'.SAC';
            self.traces.append(trace);
        return;
    
    def PlotAziStreams(self, ampfactor=0.05, title='', ax=plt.subplot(), targetDT=0.1):
        ymax=-999.;
        ymin=999.;
        
        for trace in self.traces:
            downsamplefactor=int(targetDT/trace.stats.delta)
            # trace.decimate(factor=downsamplefactor, no_filter=True);
            dt=trace.stats.delta;
            time=dt*np.arange(trace.stats.npts);
            
            yvalue=trace.data*ampfactor;
            dist=float(trace.stats.sac.dist)
            ax.plot(time, yvalue+dist, '-k', lw=1);
            tfill=time[yvalue>0];
            yfill=(yvalue+dist)[yvalue>0];
            ax.fill_between(tfill, dist, yfill, color='blue', linestyle='--', lw=0.);
            tfill=time[yvalue<0];
            yfill=(yvalue+dist)[yvalue<0];
            ax.fill_between(tfill, dist, yfill, color='red', linestyle='--', lw=0.);
            pos='lon: '+str(trace.stats.sac.stlo)+' lat: '+str(trace.stats.sac.stla)
            # plt.text(5, dist, pos)
            ymin=min(ymin, dist)
            ymax=max(ymax, dist)
            Valuemax=(npr.evaluate('abs(yvalue/ampfactor)')).max()
            # plt.text(time.max()-100, dist, str(Valuemax)+' nm')
        ymin=ymin-50;
        ymax=ymax+50;
        plt.axis([0., time.max(), ymin, ymax])
        plt.xlabel('Time(sec)');
        plt.ylabel('Distance(km)')
        plt.title(title);
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        # fig.savefig(outdir+'/ses3d_ST.pdf', orientation='landscape', format='pdf')
        return;

def StationSES3D2SAC(STA, datadir, outdir, scaling=1e9, VminPadding=-1, integrate=True, delta=0.1):
    STA.SES3D2SAC( datadir=datadir, outdir=outdir, scaling=scaling,\
            VminPadding=VminPadding, integrate=integrate, delta=delta)
    return;


    
        
def GaussianFilter(indata, fcenter, df, fhlen=0.008):
    npts=indata.size
    Ns=1<<(npts-1).bit_length()
    nhalf=Ns/2+1
    fmax=(nhalf-1)*df
    if fcenter>fmax:
        fcenter=fmax
    alpha = -0.5/(fhlen*fhlen)
    F=np.arange(Ns)*df
    gauamp = F - fcenter
    sf=npr.evaluate('exp(alpha*gauamp**2)')
    sp, Ns=FFTW(indata, direction='forward')
    filtered_sp=npr.evaluate('sf*sp')
    filtered_seis, Ns=FFTW(filtered_sp, direction='backward')
    filtered_seis=filtered_seis[:npts].real
    return filtered_seis

def FFTW(indata, direction, flags=['estimate']):
    """
    FFTW: a function utilizes fftw, a extremely fast library to do FFT computation
    Functions that using this function:
        noisetrace.GaussianFilter()
    """
    npts=indata.size
    Ns=1<<(npts-1).bit_length()
    INput = np.zeros((Ns), dtype=complex)
    OUTput = np.zeros((Ns), dtype=complex)
    fftw = fftw3.Plan(INput, OUTput, direction=direction, flags=flags)
    INput[:npts]=indata
    fftw()
    nhalf=Ns/2+1
    if direction == 'forward':
        OUTput[nhalf:]=0
        OUTput[0]/=2
        OUTput[nhalf-1]=OUTput[nhalf-1].real+0.j
    if direction =='backward':
        OUTput=2*OUTput/Ns
    return OUTput, Ns


#########################################################################
# define seismogram class for ses3d
#########################################################################

class ses3d_seismogram(object):
    
    def __init__(self):
    
        self.nt=0.0
        self.dt=0.0
        self.time=np.array([])
        self.rx=0.0
        self.ry=0.0
        self.rz=0.0
        self.sx=0.0
        self.sy=0.0
        self.sz=0.0
        self.trace_x=np.array([])
        self.trace_y=np.array([])
        self.trace_z=np.array([])
        self.integrate=True
    
    #########################################################################
    # read seismograms
    #########################################################################
    
    def read(self, directory, staname, integrate=True, fmin=0.01, fmax=1):
        """ read seismogram
        read(directory,station_name,integrate)
      
        directory: directory where seismograms are located
        station_name: name of the station, without '.x', '.y' or '.z'
        integrate: integrate original velocity seismograms to displacement seismograms (important for adjoint sources)
        """
        prefix=staname+'.___';
        self.integrate=integrate;
        if not ( os.path.isfile(directory+'/'+prefix+'.x') and os.path.isfile(directory+'/'+prefix+'.y')\
                and os.path.isfile(directory+'/'+prefix+'.z') ):
            print 'No output for ',staname;
            return False
        # open files ====================================================
        fx=open(directory+'/'+prefix+'.x','r')
        fy=open(directory+'/'+prefix+'.y','r')
        fz=open(directory+'/'+prefix+'.z','r')
        # read content ==================================================
        fx.readline()
        self.nt=int(fx.readline().strip().split('=')[1])
        self.dt=float(fx.readline().strip().split('=')[1])
        fx.readline()
        line=fx.readline().strip().split('=')
        self.rx=float(line[1].split('y')[0])
        self.ry=float(line[2].split('z')[0])
        self.rz=float(line[3])
        print 'receiver: colatitude={} deg, longitude={} deg, depth={} m'.format(self.rx,self.ry,self.rz)
        fx.readline()
        line=fx.readline().strip().split('=')
        self.sx=float(line[1].split('y')[0])
        self.sy=float(line[2].split('z')[0])
        self.sz=float(line[3])
        # print 'source: colatitude={} deg, longitude={} deg, depth={} m'.format(self.sx,self.sy,self.sz)
        for k in range(7):
            fy.readline()
            fz.readline()
        self.trace_x=np.empty(self.nt,dtype=np.float64)
        self.trace_y=np.empty(self.nt,dtype=np.float64)
        self.trace_z=np.empty(self.nt,dtype=np.float64)
      
        for k in range(self.nt):
            self.trace_x[k]=float(fx.readline().strip())
            self.trace_y[k]=float(fy.readline().strip())
            self.trace_z[k]=float(fz.readline().strip())
      
        self.time=np.linspace(0,self.nt*self.dt,self.nt)
      
        # integrate to displacement seismograms =========================
        if integrate==True:
            self.trace_x=np.cumsum(self.trace_x)*self.dt;
            self.trace_y=np.cumsum(self.trace_y)*self.dt;
            self.trace_z=np.cumsum(self.trace_z)*self.dt;
            # self.bandpass(fmin=fmin,fmax=fmax);
        # close files ===================================================
        fx.close()
        fy.close()
        fz.close()
        return True;
      #########################################################################
      # plot seismograms
      #########################################################################

    def plot(self,scaling=1.0):
        """ plot seismograms
        plot(scaling)
        scaling: scaling factor for the seismograms
        """
        plt.subplot(311)
        plt.plot(self.t,scaling*self.trace_x,'k')
        plt.grid(True)
        plt.xlabel('time [s]')
      
        if self.integrate==True:
            plt.ylabel(str(scaling)+'*u_theta [m]')
        else:
            plt.ylabel(str(scaling)+'*v_theta [m/s]')
      
        plt.subplot(312)
        plt.plot(self.t,scaling*self.trace_y,'k')
        plt.grid(True)
        plt.xlabel('time [s]')
      
        if self.integrate==True:
            plt.ylabel(str(scaling)+'*u_phi [m]')
        else:
            plt.ylabel(str(scaling)+'*v_phi [m/s]')
      
        plt.subplot(313)
        plt.plot(self.t,scaling*self.trace_z,'k')
        plt.grid(True)
        plt.xlabel('time [s]')
      
        if self.integrate==True:
            plt.ylabel(str(scaling)+'*u_r [m]')
        else:
            plt.ylabel(str(scaling)+'*v_r [m/s]')
      
        plt.show()
    
    #########################################################################
    # filter
    #########################################################################
    
    def bandpass(self,fmin,fmax):
        """
        bandpass(fmin,fmax)
        Apply a zero-phase bandpass to all traces. 
        """
        self.trace_x=flt.bandpass(self.trace_x,fmin,fmax,1.0/self.dt,corners=2,zerophase=True)
        self.trace_y=flt.bandpass(self.trace_y,fmin,fmax,1.0/self.dt,corners=2,zerophase=True)
        self.trace_z=flt.bandpass(self.trace_z,fmin,fmax,1.0/self.dt,corners=2,zerophase=True)
    
    def Convert2ObsStream(self, stacode, network, VminPadding=-1, scaling=1e9):
        ST=obspy.core.Stream();
        stla=90.-self.rx;
        stlo=self.ry;
        evla=90.-self.sx;
        evlo=self.sy;
        evdp=self.sz/1000.;
        dist, az, baz=obsGeo.gps2DistAzimuth(evla, evlo, stla, stlo); # distance is in m
        dist=dist/1000.;
        Ntmax=-1;
        Nsnap=26000;
        L=self.trace_x.size;

        if VminPadding>0:
            Ntmax=int(dist/VminPadding/self.dt);
            if Ntmax>Nsnap:
                Ntmax=-1;
            else:
                Ntmax=Nsnap;
        paddingflag=False;
        if Ntmax>0 and L>Ntmax:
            max1=np.abs(self.trace_z[:Ntmax]).max();
            # index_max2=np.abs(self.trace_z[Ntmax:]).argmax();
            max2=np.abs(self.trace_z[Ntmax:]).max();
            if max2>0.1*max1:
                paddingflag=True;
        # Initialization
        tr_x=obspy.core.trace.Trace();
        tr_y=obspy.core.trace.Trace();
        tr_z=obspy.core.trace.Trace();
        # N component
        tr_x.stats['sac']={};
            
        if paddingflag==True:
            print 'Do padding for:', stlo, stla, dist, Ntmax, L
            Xdata=self.trace_x[:Ntmax]*(-1.0)*scaling;
            tr_x.data=np.pad(Xdata,(0,L-Ntmax,),mode='linear_ramp',end_values=(0) );
            tr_x.stats.sac.kuser2='padded';
        else:
            tr_x.data=self.trace_x*(-1.0)*scaling;
        tr_x.stats.channel='BXN';
        tr_x.stats.delta=self.dt;
        tr_x.stats.station=stacode; # tr.id will be automatically assigned once given stacode and network
        tr_x.stats.network=network;
        
        tr_x.stats.sac.dist=dist;
        tr_x.stats.sac.az=az;  
        tr_x.stats.sac.baz=baz;
        tr_x.stats.sac.idep=6; # Displacement in nm
        tr_x.stats.sac.evlo=evlo;
        tr_x.stats.sac.evla=evla;
        tr_x.stats.sac.stlo=stlo;
        tr_x.stats.sac.stla=stla;
        tr_x.stats.sac.evdp=evdp;
        tr_x.stats.sac.kuser1='ses3d_4';
        # Z component
        tr_z.stats['sac']={};
        if paddingflag==True:
            Zdata=self.trace_z[:Ntmax]*scaling;
            tr_z.data=np.pad(Zdata,(0,L-Ntmax,),mode='linear_ramp',end_values=(0) );
            tr_z.stats.sac.kuser2='padded';
        else:
            tr_z.data=self.trace_z*scaling;
        # tr_z.data=self.trace_z*scaling;
        tr_z.stats.channel='BXZ';
        tr_z.stats.delta=self.dt;
        tr_z.stats.station=stacode; # tr.id will be automatically assigned once given stacode and network
        tr_z.stats.network=network;
        
        tr_z.stats.sac.dist=dist;
        tr_z.stats.sac.az=az;  
        tr_z.stats.sac.baz=baz;
        tr_z.stats.sac.idep=6; # Displacement in nm
        tr_z.stats.sac.evlo=evlo;
        tr_z.stats.sac.evla=evla;
        tr_z.stats.sac.stlo=stlo;
        tr_z.stats.sac.stla=stla;
        tr_z.stats.sac.evdp=evdp;
        tr_z.stats.sac.kuser1='ses3d_4';
        # E component
        tr_y.stats['sac']={};
        if paddingflag==True:
            Ydata=self.trace_y[:Ntmax]*scaling;
            tr_y.data=np.pad(Ydata,(0,L-Ntmax,),mode='linear_ramp',end_values=(0) );
            tr_y.stats.sac.kuser2='padded';
        else:
            tr_y.data=self.trace_y*scaling;
        # tr_y.data=self.trace_y*scaling;
        tr_y.stats.channel='BXE';
        tr_y.stats.delta=self.dt;
        tr_y.stats.station=stacode; # tr.id will be automatically assigned once given stacode and network
        tr_y.stats.network=network;
        
        tr_y.stats.sac.dist=dist;
        tr_y.stats.sac.az=az;  
        tr_y.stats.sac.baz=baz;
        tr_y.stats.sac.idep=6; # Displacement in nm
        tr_y.stats.sac.evlo=evlo;
        tr_y.stats.sac.evla=evla;
        tr_y.stats.sac.stlo=stlo;
        tr_y.stats.sac.stla=stla;
        tr_y.stats.sac.evdp=evdp;
        tr_y.stats.sac.kuser1='ses3d_4';
        
        ST.append(tr_x);
        ST.append(tr_y);
        ST.append(tr_z);
        return ST;

#- Pretty units for some components.
UNIT_DICT = {
    "vp": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "vsv": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "vsh": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    #"rho": r"$\frac{\mathrm{kg}^3}{\mathrm{m}^3}$",
    "rhoinv": r"$\frac{\mathrm{m}^3}{\mathrm{kg}^3}$",
    "vx": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "vy": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    # "vz": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "vz": r"m/s",
    "Vs": r"$\frac{\mathrm{km}}{\mathrm{sec}}$",
}
    # ==============================================================================================
    # - Class for fields (models, kernels, snapshots) in the SES3D format.
    # ==============================================================================================
class snapshotParam(object):
    """
    Snap
    """
    def __init__(self):
        #- coordinate lines
        self.lat=np.zeros(1)
        self.lon=np.zeros(1)
        self.r=np.zeros(1)
        #- rotated coordinate lines
        self.lat_rot=np.zeros(1)
        self.lon_rot=np.zeros(1)
        #- field
        self.v=np.zeros((1, 1, 1))  

        

def FFTW(indata, direction, flags=['estimate']):
    """
    FFTW: a function utilizes fftw, a extremely fast library to do FFT computation (pyfftw3 need to be installed)
    -----------------------------------------------------------------------------------------------------
    Input Parameters:
    indata      - Input data
    direction   - direction of FFT
    flags       - list of fftw-flags to be used in planning
    -----------------------------------------------------------------------------------------------------
    Functions that using this function:
        noisetrace.GaussianFilter()
    """
    npts=indata.size
    Ns=1<<(npts-1).bit_length()
    INput = np.zeros((Ns), dtype=complex)
    OUTput = np.zeros((Ns), dtype=complex)
    fftw = fftw3.Plan(INput, OUTput, direction=direction, flags=flags)
    INput[:npts]=indata
    fftw()
    nhalf=Ns/2+1
    if direction == 'forward':
        OUTput[nhalf:]=0
        OUTput[0]/=2
        OUTput[nhalf-1]=OUTput[nhalf-1].real+0.j
    if direction =='backward':
        OUTput=2*OUTput/Ns
    return OUTput, Ns      
    
    