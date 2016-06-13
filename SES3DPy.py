
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
#########################################################################
#- define submodel model class
#########################################################################
class ses3d_submodel(object):
    """ class defining an ses3d submodel
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
        
        
class ses3d_model(object):
    """ class for reading, writing, plotting and manipulating ses3d model
    """

    def __init__(self):
        """ initiate the ses3d_model class
        initiate list of submodels and read rotation_parameters.txt
        """
        self.nsubvol=0
        self.lat_min=0.0
        self.lat_max=0.0
        self.lon_min=0.0
        self.lon_max=0.0
        self.lat_centre=0.0
        self.lon_centre=0.0
        self.global_regional="global"
        self.m=[]
        #- read rotation parameters
        if not os.path.isfile('./ses3d_data/rotation_parameters.txt'):
            print './ses3d_data/rotation_parameters.txt NOT exists!'
            return;
        fid=open('./ses3d_data/rotation_parameters.txt','r')
        # fid=open('/projects/life9360/software/ses3d_r07_b/SCENARIOS/NORTH_AMERICA/TOOLS/rotation_parameters.txt','r')
        fid.readline()
        self.phi=float(fid.readline().strip())
        fid.readline()
        line=fid.readline().strip().split(' ')
        self.n=np.array([float(line[0]),float(line[1]),float(line[2])])
        fid.close()
    #########################################################################
    #- copy models
    #########################################################################
    def copy(self):
        """ Copy a model
        """
        res=ses3d_model()
        res.nsubvol=self.nsubvol
        res.lat_min=self.lat_min
        res.lat_max=self.lat_max
        res.lon_min=self.lon_min
        res.lon_max=self.lon_max
        res.lat_centre=self.lat_centre
        res.lon_centre=self.lon_centre
        res.phi=self.phi
        res.n=self.n
        res.global_regional=self.global_regional
        res.d_lon=self.d_lon
        res.d_lat=self.d_lat
        for k in np.arange(self.nsubvol):
            subvol=ses3d_submodel()
            subvol.lat=self.m[k].lat
            subvol.lon=self.m[k].lon
            subvol.r=self.m[k].r
            subvol.lat_rot=self.m[k].lat_rot
            subvol.lon_rot=self.m[k].lon_rot
            subvol.v=self.m[k].v
            res.m.append(subvol)
        return res

    #########################################################################
    #- multiplication with a scalar
    #########################################################################
    def __rmul__(self,factor):
        """ override left-multiplication of an ses3d model by a scalar factor
        """
        res=ses3d_model()
        res.nsubvol=self.nsubvol
        res.lat_min=self.lat_min
        res.lat_max=self.lat_max
        res.lon_min=self.lon_min
        res.lon_max=self.lon_max
        res.lat_centre=self.lat_centre
        res.lon_centre=self.lon_centre
        res.phi=self.phi
        res.n=self.n
        res.global_regional=self.global_regional
        res.d_lon=self.d_lon
        res.d_lat=self.d_lat
        for k in np.arange(self.nsubvol):
            subvol=ses3d_submodel()
            subvol.lat=self.m[k].lat
            subvol.lon=self.m[k].lon
            subvol.r=self.m[k].r
            subvol.lat_rot=self.m[k].lat_rot
            subvol.lon_rot=self.m[k].lon_rot
            subvol.v=factor*self.m[k].v
            res.m.append(subvol)
        return res
    #########################################################################
    #- adding two models
    #########################################################################
    def __add__(self,other_model):
        """ override addition of two ses3d models
        """
        res=ses3d_model()
        res.nsubvol=self.nsubvol
        res.lat_min=self.lat_min
        res.lat_max=self.lat_max
        res.lon_min=self.lon_min
        res.lon_max=self.lon_max
        res.lat_centre=self.lat_centre
        res.lon_centre=self.lon_centre
        res.phi=self.phi
        res.n=self.n
        res.global_regional=self.global_regional
        res.d_lon=self.d_lon
        res.d_lat=self.d_lat
        for k in np.arange(self.nsubvol):
            subvol=ses3d_submodel()
            subvol.lat=self.m[k].lat
            subvol.lon=self.m[k].lon
            subvol.r=self.m[k].r
            subvol.lat_rot=self.m[k].lat_rot
            subvol.lon_rot=self.m[k].lon_rot
            subvol.v=self.m[k].v+other_model.m[k].v
            res.m.append(subvol)
        return res
    #########################################################################
    #- read a 3D model
    #########################################################################
    def read(self, directory, filename, verbose=False):
        """ read an ses3d model from a file
        read(self,directory,filename,verbose=False):
        """
        #- read block files ====================================================
        fid_x=open(directory+'/block_x','r')
        fid_y=open(directory+'/block_y','r')
        fid_z=open(directory+'/block_z','r')
        if verbose==True:
            print 'read block files:'
            print '\t '+directory+'/block_x'
            print '\t '+directory+'/block_y'
            print '\t '+directory+'/block_z'
        dx=np.array(fid_x.read().strip().split('\n'),dtype=float)
        dy=np.array(fid_y.read().strip().split('\n'),dtype=float)
        dz=np.array(fid_z.read().strip().split('\n'),dtype=float)
        fid_x.close()
        fid_y.close()
        fid_z.close()
        #- read coordinate lines ===============================================
        self.nsubvol=int(dx[0])
        if verbose==True:
            print 'number of subvolumes: '+str(self.nsubvol)
        idx=np.ones(self.nsubvol,dtype=int)
        idy=np.ones(self.nsubvol,dtype=int)
        idz=np.ones(self.nsubvol,dtype=int)
        for k in np.arange(1, self.nsubvol, dtype=int):
            idx[k]=int(dx[idx[k-1]])+idx[k-1]+1
            idy[k]=int(dy[idy[k-1]])+idy[k-1]+1
            idz[k]=int(dz[idz[k-1]])+idz[k-1]+1    
        for k in np.arange(self.nsubvol,dtype=int):
            subvol=ses3d_submodel()
            subvol.lat=90.0-dx[(idx[k]+1):(idx[k]+1+dx[idx[k]])]
            subvol.lon=dy[(idy[k]+1):(idy[k]+1+dy[idy[k]])]
            subvol.r  =dz[(idz[k]+1):(idz[k]+1+dz[idz[k]])]
            self.m.append(subvol)
        #- compute rotated version of the coordinate lines ====================
        if self.phi!=0.0:
            for k in np.arange(self.nsubvol,dtype=int):
                nx=len(self.m[k].lat)
                ny=len(self.m[k].lon)
                self.m[k].lat_rot=np.zeros([nx,ny])
                self.m[k].lon_rot=np.zeros([nx,ny])
                for idx in np.arange(nx):
                    for idy in np.arange(ny):
                      self.m[k].lat_rot[idx,idy],self.m[k].lon_rot[idx,idy]\
                        =rotate_coordinates(self.n,-self.phi,90.0-self.m[k].lat[idx],self.m[k].lon[idy])
                      self.m[k].lat_rot[idx,idy]=90.0-self.m[k].lat_rot[idx,idy]
        else:
            for k in np.arange(self.nsubvol,dtype=int):
                self.m[k].lat_rot, self.m[k].lon_rot=np.meshgrid(self.m[k].lat,self.m[k].lon)
                self.m[k].lat_rot=self.m[k].lat_rot.T
                self.m[k].lon_rot=self.m[k].lon_rot.T
        #- read model volume ==================================================
        fid_m=open(directory+'/'+filename,'r')
        if verbose==True:
            print 'read model file: '+directory+'/'+filename
        v=np.array(fid_m.read().strip().split('\n'),dtype=float)
        fid_m.close()
        #- assign values ======================================================
        idx=1
        
        for k in np.arange(self.nsubvol):
            n=int(v[idx])
            nx=len(self.m[k].lat)-1
            ny=len(self.m[k].lon)-1
            nz=len(self.m[k].r)-1
            # print idx, n, nx, ny, nz
            self.m[k].v=v[(idx+1):(idx+1+n)].reshape(nx,ny,nz)
            idx=idx+n+1
        #- decide on global or regional model==================================
        self.lat_min=90.0
        self.lat_max=-90.0
        self.lon_min=180.0
        self.lon_max=-180.0
        for k in np.arange(self.nsubvol):
            if np.min(self.m[k].lat_rot) < self.lat_min: self.lat_min = np.min(self.m[k].lat_rot)
            if np.max(self.m[k].lat_rot) > self.lat_max: self.lat_max = np.max(self.m[k].lat_rot)
            if np.min(self.m[k].lon_rot) < self.lon_min: self.lon_min = np.min(self.m[k].lon_rot)
            if np.max(self.m[k].lon_rot) > self.lon_max: self.lon_max = np.max(self.m[k].lon_rot)
        if ((self.lat_max-self.lat_min) > 90.0 or (self.lon_max-self.lon_min) > 90.0):
            self.global_regional = "global"
            self.lat_centre = (self.lat_max+self.lat_min)/2.0
            self.lon_centre = (self.lon_max+self.lon_min)/2.0
        else:
            self.global_regional = "regional"
            self.d_lat=5.0
            self.d_lon=5.0
        return;
    #########################################################################
    #- write a 3D model to a file
    #########################################################################
    def write(self, directory, filename, verbose=False):
        """ write ses3d model to a file
        Output format:
        ==============================================================
        No. of subvolume
        total No. for subvolume 1
        ... (data in nx, ny, nz)
        total No. for subvolume 2
        ...
        ==============================================================
        write(self, directory, filename, verbose=False)
        """
        if not os.path.isdir(directory):
            os.makedirs(directory)
        fid_m=open(directory+'/'+filename,'w')
        if verbose==True:
            print 'write to file '+directory+'/'+filename
        fid_m.write(str(self.nsubvol)+'\n')
        for k in np.arange(self.nsubvol):
            nx=len(self.m[k].lat)-1
            ny=len(self.m[k].lon)-1
            nz=len(self.m[k].r)-1
            fid_m.write(str(nx*ny*nz)+'\n')
            for idx in np.arange(nx):
                for idy in np.arange(ny):
                    for idz in np.arange(nz):
                        fid_m.write(str(self.m[k].v[idx,idy,idz])+'\n')
        fid_m.close()

    #########################################################################
    #- Compute the L2 norm.
    #########################################################################
    def norm(self):
        N=0.0
        #- Loop over subvolumes. ----------------------------------------------
        for n in np.arange(self.nsubvol):
            #- Size of the array.
            nx=len(self.m[n].lat)-1
            ny=len(self.m[n].lon)-1
            nz=len(self.m[n].r)-1
            #- Compute volume elements.
            dV=np.zeros(np.shape(self.m[n].v))
            theta=(90.0-self.m[n].lat)*np.pi/180.0
            dr=self.m[n].r[1]-self.m[n].r[0]
            dphi=(self.m[n].lon[1]-self.m[n].lon[0])*np.pi/180.0
            dtheta=theta[1]-theta[0]
            for idx in np.arange(nx):
                  for idy in np.arange(ny):
                      for idz in np.arange(nz):
                          dV[idx,idy,idz]=theta[idx]*(self.m[n].r[idz])**2
            dV=dr*dtheta*dphi*dV
            #- Integrate.
            N+=np.sum(dV*(self.m[n].v)**2)
        #- Finish. ------------------------------------------------------------
        return np.sqrt(N)
    #########################################################################
    #- Remove the upper percentiles of a model.
    #########################################################################
    def clip_percentile(self, percentile):
        """
        Clip the upper percentiles of the model. Particularly useful to remove the singularities in sensitivity kernels.
        """
        #- Loop over subvolumes to find the percentile.
        percentile_list=[]
        for n in np.arange(self.nsubvol):
            percentile_list.append(np.percentile(np.abs(self.m[n].v), percentile))
        percent=np.max(percentile_list)
        #- Clip the values above the percentile.
        for n in np.arange(self.nsubvol):
            idx=np.nonzero(np.greater(np.abs(self.m[n].v),percent))
            self.m[n].v[idx]=np.sign(self.m[n].v[idx])*percent
    #########################################################################
    #- Apply horizontal smoothing.
    #########################################################################
    def smooth_horizontal(self, sigma, filter_type='gauss'):
        """
        smooth_horizontal(self,sigma,filter='gauss')
        Experimental function for smoothing in horizontal directions.
        filter_type: gauss (Gaussian smoothing), neighbour (average over neighbouring cells)
        sigma: filter width (when filter_type='gauss') or iterations (when filter_type='neighbour')
        WARNING: Currently, the smoothing only works within each subvolume. The problem 
        of smoothing across subvolumes without having excessive algorithmic complexity 
        and with fast compute times, awaits resolution ... .
        """
        #- Loop over subvolumes.---------------------------------------------------
        for n in np.arange(self.nsubvol):
            v_filtered=self.m[n].v
            #- Size of the array.
            nx=len(self.m[n].lat)-1
            ny=len(self.m[n].lon)-1
            nz=len(self.m[n].r)-1
            #- Gaussian smoothing. --------------------------------------------------
            if filter_type=='gauss':
                #- Estimate element width.
                r=np.mean(self.m[n].r)
                dx=r*np.pi*(self.m[n].lat[0]-self.m[n].lat[1])/180.0
                #- Colat and lon fields for the small Gaussian.
                dn=3*np.ceil(sigma/dx)
                nx_min=np.round(float(nx)/2.0)-dn
                nx_max=np.round(float(nx)/2.0)+dn
                ny_min=np.round(float(ny)/2.0)-dn
                ny_max=np.round(float(ny)/2.0)+dn
                lon,colat=np.meshgrid(self.m[n].lon[ny_min:ny_max],90.0-self.m[n].lat[nx_min:nx_max])
                colat=np.pi*colat/180.0
                lon=np.pi*lon/180.0
                #- Volume element.
                dy=r*np.pi*np.sin(colat)*(self.m[n].lon[1]-self.m[n].lon[0])/180.0
                dV=dx*dy
                #- Unit vector field.
                x=np.cos(lon)*np.sin(colat)
                y=np.sin(lon)*np.sin(colat)
                z=np.cos(colat)
                #- Make a Gaussian centred in the middle of the grid. -----------------
                i=np.round(float(nx)/2.0)-1
                j=np.round(float(ny)/2.0)-1
                colat_i=np.pi*(90.0-self.m[n].lat[i])/180.0
                lon_j=np.pi*self.m[n].lon[j]/180.0
                x_i=np.cos(lon_j)*np.sin(colat_i)
                y_j=np.sin(lon_j)*np.sin(colat_i)
                z_k=np.cos(colat_i)
                #- Compute the Gaussian.
                G=x*x_i+y*y_j+z*z_k
                G=G/np.max(np.abs(G))
                G=r*np.arccos(G)
                G=np.exp(-0.5*G**2/sigma**2)/(2.0*np.pi*sigma**2)
                #- Move the Gaussian across the field. --------------------------------
                for i in np.arange(dn+1,nx-dn-1):
                    for j in np.arange(dn+1,ny-dn-1):
                        for k in np.arange(nz):
                            v_filtered[i,j,k]=np.sum(self.m[n].v[i-dn:i+dn,j-dn:j+dn,k]*G*dV)
            #- Smoothing by averaging over neighbouring cells. ----------------------
            elif filter_type=='neighbour':
                for iteration in np.arange(int(sigma)):
                    for i in np.arange(1,nx-1):
                        for j in np.arange(1,ny-1):
                            v_filtered[i,j,:]=(self.m[n].v[i,j,:]+self.m[n].v[i+1,j,:]+self.m[n].v[i-1,j,:]\
                                +self.m[n].v[i,j+1,:]+self.m[n].v[i,j-1,:])/5.0
            self.m[n].v=v_filtered
        return;
        
    
    #########################################################################
    #- Apply horizontal smoothing with adaptive smoothing length.
    #########################################################################
    def smooth_horizontal_adaptive(self,sigma):
        #- Find maximum smoothing length. -------------------------------------
        sigma_max=[]
        for n in np.arange(self.nsubvol):
            sigma_max.append(np.max(sigma.m[n].v))
        #- Loop over subvolumes.-----------------------------------------------
        for n in np.arange(self.nsubvol):
            #- Size of the array.
            nx=len(self.m[n].lat)-1
            ny=len(self.m[n].lon)-1
            nz=len(self.m[n].r)-1
            #- Estimate element width.
            r=np.mean(self.m[n].r)
            dx=r*np.pi*(self.m[n].lat[0]-self.m[n].lat[1])/180.0
            #- Colat and lon fields for the small Gaussian.
            dn=2*np.round(sigma_max[n]/dx)
            nx_min=np.round(float(nx)/2.0)-dn
            nx_max=np.round(float(nx)/2.0)+dn
            ny_min=np.round(float(ny)/2.0)-dn
            ny_max=np.round(float(ny)/2.0)+dn
            lon,colat=np.meshgrid(self.m[n].lon[ny_min:ny_max],90.0-self.m[n].lat[nx_min:nx_max],dtype=float)
            colat=np.pi*colat/180.0
            lon=np.pi*lon/180.0
            #- Volume element.
            dy=r*np.pi*np.sin(colat)*(self.m[n].lon[1]-self.m[n].lon[0])/180.0
            dV=dx*dy
            #- Unit vector field.
            x=np.cos(lon)*np.sin(colat)
            y=np.sin(lon)*np.sin(colat)
            z=np.cos(colat)
            #- Make a Gaussian centred in the middle of the grid. ---------------
            i=np.round(float(nx)/2.0)-1
            j=np.round(float(ny)/2.0)-1
            colat_i=np.pi*(90.0-self.m[n].lat[i])/180.0
            lon_j=np.pi*self.m[n].lon[j]/180.0
            x_i=np.cos(lon_j)*np.sin(colat_i)
            y_j=np.sin(lon_j)*np.sin(colat_i)
            z_k=np.cos(colat_i)
            #- Distance from the central point.
            G=x*x_i+y*y_j+z*z_k
            G=G/np.max(np.abs(G))
            G=r*np.arccos(G)
            #- Move the Gaussian across the field. ------------------------------
            v_filtered=self.m[n].v
            for i in np.arange(dn+1,nx-dn-1):
                for j in np.arange(dn+1,ny-dn-1):
                    for k in np.arange(nz):
                        #- Compute the actual Gaussian.
                        s=sigma.m[n].v[i,j,k]
                        if (s>0):
                            GG=np.exp(-0.5*G**2/s**2)/(2.0*np.pi*s**2)
                            #- Apply filter.
                            v_filtered[i,j,k]=np.sum(self.m[n].v[i-dn:i+dn,j-dn:j+dn,k]*GG*dV)
            self.m[n].v=v_filtered
    #########################################################################
    #- Compute relaxed velocities from velocities at 1 s reference period.
    #########################################################################
    def ref2relax(self, datadir, qmodel='cem', nrelax=3):
        """
        ref2relax(qmodel='cem', nrelax=3)
        Assuming that the current velocity model is given at the reference period 1 s, 
        ref2relax computes the relaxed velocities. They may then be written to a file.
        For this conversion, the relaxation parameters from the relax file are taken.
        Currently implemented Q models (qmodel): cem, prem, ql6 . 
        nrelax is the number of relaxation mechnisms.
        """
        #- Read the relaxation parameters from the relax file. ----------------
        tau_p=np.zeros(nrelax)
        D_p=np.zeros(nrelax)
        relaxfname=datadir+'/'+'relax'
        fid=open(relaxfname,'r')
        fid.readline()
        for n in range(nrelax):
            tau_p[n]=float(fid.readline().strip())
        fid.readline()
        for n in range(nrelax):
            D_p[n]=float(fid.readline().strip())
        fid.close()
        #- Loop over subvolumes. ----------------------------------------------
        for k in np.arange(self.nsubvol):
            nx=len(self.m[k].lat)-1
            ny=len(self.m[k].lon)-1
            nz=len(self.m[k].r)-1
            #- Loop over radius within the subvolume. ---------------------------
            for idz in np.arange(nz):
                #- Compute Q. -----------------------------------------------------
                if qmodel=='cem':
                  Q=q.q_cem(self.m[k].r[idz])
                elif qmodel=='ql6':
                  Q=q.q_ql6(self.m[k].r[idz])
                elif qmodel=='prem':
                  Q=q.q_prem(self.m[k].r[idz])
                #- Compute A and B for the reference period of 1 s. ---------------
                A=1.0
                B=0.0
                w=2.0*np.pi
                tau=1.0/Q
                for n in range(nrelax):
                    A+=tau*D_p[n]*(w**2)*(tau_p[n]**2)/(1.0+(w**2)*(tau_p[n]**2))
                    B+=tau*D_p[n]*w*tau_p[n]/(1.0+(w**2)*(tau_p[n]**2))
                conversion_factor=(A+np.sqrt(A**2+B**2))/(A**2+B**2)
                conversion_factor=np.sqrt(0.5*conversion_factor)
                #- Correct velocities. --------------------------------------------
                self.m[k].v[:,:,idz]=conversion_factor*self.m[k].v[:,:,idz]
        return;
    #########################################################################
    #- convert to vtk format
    #########################################################################
    def convert_to_vtk(self, directory, filename, verbose=False):
        """ convert ses3d model to vtk format for plotting with Paraview, VisIt, ... .
        convert_to_vtk(self,directory,filename,verbose=False):
        """
        Rref=6471.;
        #- preparatory steps
        nx=np.zeros(self.nsubvol, dtype=int)
        ny=np.zeros(self.nsubvol, dtype=int)
        nz=np.zeros(self.nsubvol, dtype=int)
        N=0
        for n in np.arange(self.nsubvol):
            nx[n]=len(self.m[n].lat)
            ny[n]=len(self.m[n].lon)
            nz[n]=len(self.m[n].r)
            N=N+nx[n]*ny[n]*nz[n]
        #- open file and write header
        fid=open(directory+'/'+filename,'w')
        if verbose==True:
            print 'write to file '+directory+filename
        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        #- write grid points
        fid.write('POINTS '+str(N)+' float\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing grid points for subvolume '+str(n)
            for i in np.arange(nx[n]):
                for j in np.arange(ny[n]):
                    for k in np.arange(nz[n]):
                        theta=90.0-self.m[n].lat[i]
                        phi=self.m[n].lon[j]
                        #- rotate coordinate system
                        if self.phi!=0.0:
                            theta,phi=rotate_coordinates(self.n,-self.phi,theta,phi)
                            #- transform to cartesian coordinates and write to file
                        theta=theta*np.pi/180.0
                        phi=phi*np.pi/180.0
                        r=self.m[n].r[k]/Rref; ### !!!
                        x=r*np.sin(theta)*np.cos(phi);
                        y=r*np.sin(theta)*np.sin(phi);
                        z=r*np.cos(theta);
                        fid.write(str(x)+' '+str(y)+' '+str(z)+'\n')
        #- write connectivity
        n_cells=0
        for n in np.arange(self.nsubvol):
            n_cells=n_cells+(nx[n]-1)*(ny[n]-1)*(nz[n]-1)
        fid.write('\n')
        fid.write('CELLS '+str(n_cells)+' '+str(9*n_cells)+'\n')
        count=0
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing conectivity for subvolume '+str(n)
            for i in np.arange(1,nx[n]):
                for j in np.arange(1,ny[n]):
                    for k in np.arange(1,nz[n]):
                        a=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        b=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n] 
                        c=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        d=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]
                        e=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]-1
                        f=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]
                        g=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]-1
                        h=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]
                        fid.write('8 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)+' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')
            count=count+nx[n]*ny[n]*nz[n]
        #- write cell types
        fid.write('\n')
        fid.write('CELL_TYPES '+str(n_cells)+'\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing cell types for subvolume '+str(n)
            for i in np.arange(nx[n]-1):
                for j in np.arange(ny[n]-1):
                    for k in np.arange(nz[n]-1):
                        fid.write('11\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS scalars float\n')
        fid.write('LOOKUP_TABLE mytable\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing data for subvolume '+str(n)
            idx=np.arange(nx[n])
            idx[nx[n]-1]=nx[n]-2
            idy=np.arange(ny[n])
            idy[ny[n]-1]=ny[n]-2
            idz=np.arange(nz[n])
            idz[nz[n]-1]=nz[n]-2
            for i in idx:
                for j in idy:
                    for k in idz:
                        fid.write(str(self.m[n].v[i,j,k])+'\n')
        #- clean up
        fid.close()
        return;
    
    
    def convert_to_vtk_depth(self, depth, directory, filename, verbose=False):
        """ convert ses3d model to vtk format for plotting with Paraview, VisIt, ... .
        convert_to_vtk(self,directory,filename,verbose=False):
        """
        Rref=6471.;
        #- preparatory steps
        nx=np.zeros(self.nsubvol, dtype=int)
        ny=np.zeros(self.nsubvol, dtype=int)
        nz=np.zeros(self.nsubvol, dtype=int)
        N=0
        for n in np.arange(self.nsubvol):
            nx[n]=len(self.m[n].lat)
            ny[n]=len(self.m[n].lon)
            nz[n]=len(self.m[n].r)
            N=N+nx[n]*ny[n]*nz[n]
        #- open file and write header
        fid=open(directory+'/'+filename,'w')
        if verbose==True:
            print 'write to file '+directory+filename
        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        #- write grid points
        fid.write('POINTS '+str(N)+' float\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing grid points for subvolume '+str(n)
            for i in np.arange(nx[n]):
                for j in np.arange(ny[n]):
                    for k in np.arange(nz[n]):
                        theta=90.0-self.m[n].lat[i]
                        phi=self.m[n].lon[j]
                        #- rotate coordinate system
                        if self.phi!=0.0:
                            theta,phi=rotate_coordinates(self.n,-self.phi,theta,phi)
                            #- transform to cartesian coordinates and write to file
                        theta=theta*np.pi/180.0
                        phi=phi*np.pi/180.0
                        r=self.m[n].r[k]/Rref; ### !!!
                        x=r*np.sin(theta)*np.cos(phi);
                        y=r*np.sin(theta)*np.sin(phi);
                        z=r*np.cos(theta);
                        fid.write(str(x)+' '+str(y)+' '+str(z)+'\n')
        #- write connectivity
        n_cells=0
        for n in np.arange(self.nsubvol):
            n_cells=n_cells+(nx[n]-1)*(ny[n]-1)*(nz[n]-1)
        fid.write('\n')
        fid.write('CELLS '+str(n_cells)+' '+str(9*n_cells)+'\n')
        count=0
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing conectivity for subvolume '+str(n)
            for i in np.arange(1,nx[n]):
                for j in np.arange(1,ny[n]):
                    for k in np.arange(1,nz[n]):
                        a=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        b=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n] 
                        c=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        d=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]
                        e=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]-1
                        f=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]
                        g=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]-1
                        h=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]
                        fid.write('8 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)+' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')
            count=count+nx[n]*ny[n]*nz[n]
        #- write cell types
        fid.write('\n')
        fid.write('CELL_TYPES '+str(n_cells)+'\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing cell types for subvolume '+str(n)
            for i in np.arange(nx[n]-1):
                for j in np.arange(ny[n]-1):
                    for k in np.arange(nz[n]-1):
                        fid.write('11\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS scalars float\n')
        fid.write('LOOKUP_TABLE mytable\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing data for subvolume '+str(n)
            idx=np.arange(nx[n])
            idx[nx[n]-1]=nx[n]-2
            idy=np.arange(ny[n])
            idy[ny[n]-1]=ny[n]-2
            idz=np.arange(nz[n])
            idz[nz[n]-1]=nz[n]-2
            for i in idx:
                for j in idy:
                    for k in idz:
                        fid.write(str(self.m[n].v[i,j,k])+'\n')
        #- clean up
        fid.close()
        return;
    #########################################################################
    #- plot horizontal slices
    #########################################################################
    def plot_slice(self, depth, min_val_plot=None, max_val_plot=None, colormap='tomo', res='i', save_under=None, verbose=False, \
                   mapfactor=2, maxlon=None, minlon=None, maxlat=None, minlat=None, geopolygons=[]):
        """ plot horizontal slices through an ses3d model
        plot_slice(self,depth,colormap='tomo',res='i',save_under=None,verbose=False)
        depth=depth in km of the slice
        colormap='tomo','mono'
        res=resolution of the map, admissible values are: c, l, i, h f
        save_under=save figure as *.png with the filename "save_under". Prevents plotting of the slice.
        """
        radius=6371.0-depth
        # ax=plt.subplot(111)
        #- set up a map and colourmap -----------------------------------------
        if self.global_regional=='regional':
            m=Basemap(projection='merc',llcrnrlat=self.lat_min,urcrnrlat=self.lat_max,llcrnrlon=self.lon_min,urcrnrlon=self.lon_max,lat_ts=20,resolution=res)
            m.drawparallels(np.arange(self.lat_min,self.lat_max,self.d_lon),labels=[1,0,0,1])
            m.drawmeridians(np.arange(self.lon_min,self.lon_max,self.d_lat),labels=[1,0,0,1])
        elif self.global_regional=='global':
            self.lat_centre = (self.lat_max+self.lat_min)/2.0
            self.lon_centre = (self.lon_max+self.lon_min)/2.0
            m=Basemap(projection='ortho',lon_0=self.lon_centre, lat_0=self.lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif self.global_regional=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=self.lon_min, lat_0=self.lat_min, resolution='l')
            m = Basemap(projection='ortho', lon_0=self.lon_min,lat_0=self.lat_min, resolution=res,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries()
        # m.drawmapboundary(fill_color=[1.0,1.0,1.0])
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        # m.drawlsmask(land_color='0.8', ocean_color='#99ffff')
        m.drawmapboundary(fill_color="white")
        if colormap=='tomo':
            my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0],\
                0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], \
                0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif colormap=='mono':
            my_colormap=make_colormap({0.0:[1.0,1.0,1.0], 0.15:[1.0,1.0,1.0], 0.85:[0.0,0.0,0.0], 1.0:[0.0,0.0,0.0]})
        #- loop over subvolumes to collect information ------------------------
        x_list=[]
        y_list=[]
        idz_list=[]
        N_list=[]
        for k in np.arange(self.nsubvol):
            nx=len(self.m[k].lat)
            ny=len(self.m[k].lon)
            r=self.m[k].r
            #- collect subvolumes within target depth
            if (max(r)>=radius) & (min(r)<radius):
                N_list.append(k)
                r=r[0:len(r)-1]
                idz=min(np.where(min(np.abs(r-radius))==np.abs(r-radius))[0])
                if idz==len(r): idz-=idz
                idz_list.append(idz)
                if verbose==True:
                    print 'true plotting depth: '+str(6371.0-r[idz])+' km'
                x,y=m(self.m[k].lon_rot[0:nx-1,0:ny-1]+0.25,self.m[k].lat_rot[0:nx-1,0:ny-1]-0.25)
                # x,y=m(self.m[k].lon_rot[0:nx-1,0:ny-1],self.m[k].lat_rot[0:nx-1,0:ny-1])
                x_list.append(x)
                y_list.append(y)
        #- make a (hopefully) intelligent colour scale ------------------------
        if min_val_plot is None:
            if len(N_list)>0:
                #- compute some diagnostics
                min_list=[]
                max_list=[]
                percentile_list=[]
                for k in np.arange(len(N_list)):
                    min_list.append(np.min(self.m[N_list[k]].v[:,:,idz_list[k]]))
                    max_list.append(np.max(self.m[N_list[k]].v[:,:,idz_list[k]]))
                    percentile_list.append(np.percentile(np.abs(self.m[N_list[k]].v[:,:,idz_list[k]]),99.0))
                minval=np.min(min_list)
                maxval=np.max(max_list)
                percent=np.max(percentile_list)
                #- min and max roughly centred around zero
                if (minval*maxval<0.0):
                    max_val_plot=percent
                    min_val_plot=-max_val_plot
                #- min and max not centred around zero
                else:
                    max_val_plot=maxval
                    min_val_plot=minval
        if len(geopolygons)!=0:
            geopolygons.PlotPolygon(mybasemap=m);
        #- loop over subvolumes to plot ---------------------------------------
        for k in np.arange(len(N_list)):
            im=m.pcolormesh(x_list[k],y_list[k],self.m[N_list[k]].v[:,:,idz_list[k]], shading='gouraud', cmap=my_colormap,vmin=min_val_plot,vmax=max_val_plot)
            # im=m.imshow(self.m[N_list[k]].v[:,:,idz_list[k]], cmap=my_colormap,vmin=min_val_plot,vmax=max_val_plot)
          #if colormap=='mono':
            #cs=m.contour(x_list[k],y_list[k],self.m[N_list[k]].v[:,:,idz_list[k]], colors='r',linewidths=1.0)
            #plt.clabel(cs,colors='r')
        #- make a colorbar and title ------------------------------------------
        cb=m.colorbar(im,"right", size="3%", pad='2%', )
        cb.ax.tick_params(labelsize=15)
        cb.set_label('km/sec', fontsize=20, rotation=90)
        # im.ax.tick_params(labelsize=20)
        plt.title(str(depth)+' km', fontsize=30)
        
        if minlon!=None and minlat!=None and maxlon!=None and maxlat!=None:
            blon=np.arange(100)*(maxlon-minlon)/100.+minlon;
            blat=np.arange(100)*(maxlat-minlat)/100.+minlat;
            Blon=blon;
            Blat=np.ones(Blon.size)*minlat;
            x,y = m(Blon, Blat)
            m.plot(x, y, 'b-', lw=3)
            
            Blon=blon;
            Blat=np.ones(Blon.size)*maxlat;
            x,y = m(Blon, Blat)
            m.plot(x, y, 'b-', lw=3)
            
            Blon=np.ones(Blon.size)*minlon;
            Blat=blat;
            x,y = m(Blon, Blat)
            m.plot(x, y, 'b-', lw=3)
            
            Blon=np.ones(Blon.size)*maxlon;
            Blat=blat;
            x,y = m(Blon, Blat)
            m.plot(x, y, 'b-', lw=3)
        
        # if len(geopolygons)!=0:
        #     geopolygons.PlotPolygon(mybasemap=m);
        #- save image if wanted -----------------------------------------------
        # if save_under is None:
        #     plt.show()
        # else:
        #     plt.savefig(save_under+'.png', format='png', dpi=200)
        #     plt.close()
        return;
    #########################################################################
    #- plot depth to a certain threshold value
    #########################################################################
    def plot_threshold(self, val, min_val_plot, max_val_plot, res='i', colormap='tomo', verbose=False):
        """ plot depth to a certain threshold value 'val' in an ses3d model
        plot_threshold(val,min_val_plot,max_val_plot,colormap='tomo',verbose=False):
        val=threshold value
        min_val_plot, max_val_plot=minimum and maximum values of the colour scale
        colormap='tomo','mono'
        """
        #- set up a map and colourmap
        if self.global_regional=='regional':
            m=Basemap(projection='merc',llcrnrlat=self.lat_min,urcrnrlat=self.lat_max,\
                    llcrnrlon=self.lon_min,urcrnrlon=self.lon_max,lat_ts=20,resolution=res)
            m.drawparallels(np.arange(self.lat_min,self.lat_max,self.d_lon),labels=[1,0,0,1])
            m.drawmeridians(np.arange(self.lon_min,self.lon_max,self.d_lat),labels=[1,0,0,1])
        elif self.global_regional=='global':
            m=Basemap(projection='ortho',lon_0=self.lon_centre,lat_0=self.lat_centre,resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color=[1.0,1.0,1.0])
        if colormap=='tomo':
            my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], \
                0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], \
                0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif colormap=='mono':
            my_colormap=make_colormap({0.0:[1.0,1.0,1.0], 0.15:[1.0,1.0,1.0], 0.85:[0.0,0.0,0.0], 1.0:[0.0,0.0,0.0]})
        #- loop over subvolumes
        for k in np.arange(self.nsubvol):
            depth=np.zeros(np.shape(self.m[k].v[:,:,0]))
            nx=len(self.m[k].lat)
            ny=len(self.m[k].lon)
            #- find depth
            r=self.m[k].r
            r=0.5*(r[0:len(r)-1]+r[1:len(r)])
            for idx in np.arange(nx-1):
                for idy in np.arange(ny-1):
                    n=self.m[k].v[idx,idy,:]>=val
                    depth[idx,idy]=6371.0-np.max(r[n])
          #- rotate coordinate system if necessary
            lon,lat=np.meshgrid(self.m[k].lon[0:ny],self.m[k].lat[0:nx])
            if self.phi!=0.0:
                lat_rot=np.zeros(np.shape(lon),dtype=float)
                lon_rot=np.zeros(np.shape(lat),dtype=float)
                for idx in np.arange(nx):
                  for idy in np.arange(ny):
                    colat=90.0-lat[idx,idy]
                    lat_rot[idx,idy],lon_rot[idx,idy]=rotate_coordinates(self.n,-self.phi,colat,lon[idx,idy])
                    lat_rot[idx,idy]=90.0-lat_rot[idx,idy]
                lon=lon_rot
                lat=lat_rot
        #- convert to map coordinates and plot
            x,y=m(lon,lat)
            im=m.pcolor(x, y, depth, cmap=my_colormap, vmin=min_val_plot, vmax=max_val_plot)
        m.colorbar(im,"right", size="3%", pad='2%')
        plt.title('depth to '+str(val)+' km/s [km]')
        plt.show()   

 
    
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

class ses3d_fields(object):
    """
    Class for reading and plotting 3D fields defined on the SEM grid of SES3D.
    """
    def __init__(self, directory, rotationfile, setupfile, recfile='', field_type="earth_model"):
        """
        __init__(self, directory, field_type="earth_model")
        Initiate the ses3d_fields class. Read available components.Admissible field_type's currentlyare "earth_model", "velocity_snapshot", and "kernel".
        """
        self.directory = directory
        self.field_type = field_type
        #- Read available Earth model files. ------------------------------------------------------
        if field_type == "earth_model":
            self.pure_components = ["A", "B", "C", "lambda", "mu", "rhoinv", "Q"]
            self.derived_components = ["vp", "vsh", "vsv", "rho"]
        #- Read available velocity snapshots
        if field_type == "velocity_snapshot":
            self.pure_components = ["vx", "vy", "vz"]
            self.derived_components = {}
        #- Read available kernels. ----------------------------------------------------------------
        if field_type == "kernel":
            self.pure_components = ["cp", "csh", "csv", "rho", "Q_mu", "Q_kappa", "alpha_mu", "alpha_kappa"]
            self.derived_components = {}
        #- Read setup file and make coordinates
        self.setup = self.read_setup(setupfile=setupfile)
        self.make_coordinates()
        #- Read rotation parameters. --------------------------------------------------------------
        fid = open(rotationfile,'r')
        fid.readline()
        self.rotangle = float(fid.readline().strip())
        fid.readline()
        line = fid.readline().strip().split(' ')
        self.n = np.array([float(line[0]),float(line[1]),float(line[2])])
        fid.close()
        #- Read station locations, if available. --------------------------------------------------
        if os.path.exists(recfile):
            self.stations = True
            f = open(recfile)
            self.n_stations = int(f.readline())
            self.stnames = []
            self.stlats = []
            self.stlons = []
            for n in range(self.n_stations):
                self.stnames.append(f.readline().strip())
                dummy = f.readline().strip().split(' ')
                self.stlats.append(90.0-float(dummy[0]))
                self.stlons.append(float(dummy[1]))
            f.close()
        else:
            self.stations = False
        return;
    #==============================================================================================
    #- Read the setup file.
    #==============================================================================================
    def read_setup(self, setupfile):
        """
        Read the setup file to get domain geometry.
        """
        setup = {}
        #- Open setup file and read header. -------------------------------------------------------
        if not os.path.isfile(setupfile):
            raise NameError('setup file does not exists!');
        f = open(setupfile,'r')
        lines = f.readlines()[1:]
        lines = [_i.strip() for _i in lines if _i.strip()]
        #- Read computational domain. -------------------------------------------------------------
        domain = {}
        domain["theta_min"] = float(lines.pop(0).split(' ')[0]) * np.pi / 180.0
        domain["theta_max"] = float(lines.pop(0).split(' ')[0]) * np.pi / 180.0
        domain["phi_min"] = float(lines.pop(0).split(' ')[0]) * np.pi / 180.0
        domain["phi_max"] = float(lines.pop(0).split(' ')[0]) * np.pi / 180.0
        domain["z_min"] = float(lines.pop(0).split(' ')[0])
        domain["z_max"] = float(lines.pop(0).split(' ')[0])
        setup["domain"] = domain
        #- Read computational setup. --------------------------------------------------------------
        lines.pop(0)
        lines.pop(0)
        lines.pop(0)
        elements = {}
        elements["nx_global"] = int(lines.pop(0).split(' ')[0])
        elements["ny_global"] = int(lines.pop(0).split(' ')[0])
        elements["nz_global"] = int(lines.pop(0).split(' ')[0])
        setup["lpd"] = int(lines.pop(0).split(' ')[0])
        procs = {}
        procs["px"] = int(lines.pop(0).split(' ')[0])
        procs["py"] = int(lines.pop(0).split(' ')[0])
        procs["pz"] = int(lines.pop(0).split(' ')[0])
        setup["procs"] = procs
        elements["nx"] = 1 + elements["nx_global"] / procs["px"]
        elements["ny"] = 1 + elements["ny_global"] / procs["py"]
        elements["nz"] = 1 + elements["nz_global"] / procs["pz"]
        setup["elements"] = elements
        #- Clean up. ------------------------------------------------------------------------------
        f.close()
        return setup
    #==============================================================================================
    #- Make coordinate lines for each chunk.
    #==============================================================================================                                    
    def make_coordinates(self):
        """
        Make the coordinate lines for the different processor boxes.
        """
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        #- Boundaries of the processor blocks. ----------------------------------------------------
        width_theta = (self.setup["domain"]["theta_max"] - self.setup["domain"]["theta_min"]) / self.setup["procs"]["px"]
        width_phi = (self.setup["domain"]["phi_max"] - self.setup["domain"]["phi_min"]) / self.setup["procs"]["py"]
        width_z = (self.setup["domain"]["z_max"] - self.setup["domain"]["z_min"]) / self.setup["procs"]["pz"]
        boundaries_theta = np.arange(self.setup["domain"]["theta_min"],self.setup["domain"]["theta_max"]+width_theta,width_theta)
        boundaries_phi = np.arange(self.setup["domain"]["phi_min"],self.setup["domain"]["phi_max"]+width_phi,width_phi)
        boundaries_z = np.arange(self.setup["domain"]["z_min"],self.setup["domain"]["z_max"]+width_z,width_z)
        #- Make knot lines. -----------------------------------------------------------------------
        knot_x = self.get_GLL() + 1.0
        for ix in np.arange(self.setup["elements"]["nx"] - 1):
            knot_x = np.append(knot_x,self.get_GLL() + 1 + 2*(ix+1))
        knot_y = self.get_GLL() + 1.0
        for iy in np.arange(self.setup["elements"]["ny"] - 1):
            knot_y = np.append(knot_y,self.get_GLL() + 1 + 2*(iy+1))
        knot_z = self.get_GLL() + 1.0
        for iz in np.arange(self.setup["elements"]["nz"] - 1):
            knot_z = np.append(knot_z,self.get_GLL() + 1 + 2*(iz+1))
        knot_x = knot_x * width_theta / np.max(knot_x)
        knot_y = knot_y * width_phi / np.max(knot_y)
        knot_z = knot_z * width_z / np.max(knot_z)
        #- Loop over all processors. --------------------------------------------------------------
        self.theta = np.empty(shape=(n_procs,len(knot_x)))
        self.phi = np.empty(shape=(n_procs,len(knot_y)))
        self.z = np.empty(shape=(n_procs,len(knot_z)))
        p = 0
        for iz in np.arange(self.setup["procs"]["pz"]):
            for iy in np.arange(self.setup["procs"]["py"]):
                for ix in np.arange(self.setup["procs"]["px"]):
                    self.theta[p,:] = boundaries_theta[ix] + knot_x
                    self.phi[p,:] = boundaries_phi[iy] + knot_y
                    self.z[p,: :-1] = boundaries_z[iz] + knot_z
                    p += 1;
        return;
    # ==============================================================================================
    # - Get GLL points.
    # ==============================================================================================
    def get_GLL(self):
        """
        Set Gauss-Lobatto-Legendre(GLL) points for a given Lagrange polynomial degree.
        """
        if self.setup["lpd"] == 2:
            knots = np.array([-1.0, 0.0, 1.0])
        elif self.setup["lpd"] == 3:
            knots = np.array([-1.0, -0.4472135954999579, 0.4472135954999579, 1.0])
        elif self.setup["lpd"] == 4:
            knots = np.array([-1.0, -0.6546536707079772, 0.0, 0.6546536707079772, 1.0])
        elif self.setup["lpd"] == 5:
            knots = np.array([-1.0, -0.7650553239294647, -0.2852315164806451, 0.2852315164806451, 0.7650553239294647, 1.0])
        elif self.setup["lpd"] == 6:
            knots = np.array([-1.0, -0.8302238962785670, -0.4688487934707142, 0.0, 0.4688487934707142, 0.8302238962785670, 1.0])
        elif self.setup["lpd"] == 7:
            knots = np.array([-1.0, -0.8717401485096066, -0.5917001814331423,\
                -0.2092992179024789, 0.2092992179024789, 0.5917001814331423, 0.8717401485096066, 1.0])
        return knots;
    # ==============================================================================================
    # - Compose filenames.
    # ==============================================================================================
    def compose_filenames(self, component, proc_number, iteration=0):
        """
        Build filenames for the different field types.
        """
        # - Earth models. --------------------------------------------------------------------------
        if self.field_type == "earth_model":
            filename = os.path.join(self.directory, component+str(proc_number))
        # - Velocity field snapshots. --------------------------------------------------------------
        elif self.field_type == "velocity_snapshot":
            filename = os.path.join(self.directory, component+"_"+str(proc_number)+"_"+str(iteration))
        # - Sensitivity kernels. -------------------------------------------------------------------
        elif self.field_type == "kernel":
            filename = os.path.join(self.directory, "grad_"+component+"_"+str(proc_number))
        return filename;
    # ==============================================================================================
    # - Read single box.
    # ==============================================================================================
    def read_single_box(self, component, proc_number, iteration=0):
        """
        Read the field from one single processor box.
        """
        # - Shape of the Fortran binary file. ------------------------------------------------------
        shape = (self.setup["elements"]["nx"],self.setup["elements"]["ny"],\
            self.setup["elements"]["nz"],self.setup["lpd"]+1,self.setup["lpd"]+1,self.setup["lpd"]+1)
        # - Read and compute the proper components. ------------------------------------------------
        if component in self.pure_components:
            filename = self.compose_filenames(component, proc_number, iteration)
            with open(filename, "rb") as open_file:
                field = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
        elif component in self.derived_components:
            # - rho 
            if component == "rho":
                filename = self.compose_filenames("rhoinv", proc_number, 0)
                with open(filename, "rb") as open_file:
                    field = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                field = 1.0 / field
            # - vp
            if component == "vp":
                filename1 = self.compose_filenames("lambda", proc_number, 0)
                filename2 = self.compose_filenames("mu", proc_number, 0)
                filename3 = self.compose_filenames("rhoinv", proc_number, 0)
                with open(filename1, "rb") as open_file:
                    field1 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                with open(filename2, "rb") as open_file:
                    field2 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                with open(filename3, "rb") as open_file:
                    field3 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                field = np.sqrt((field1 + 2 * field2) * field3)
            # - vsh
            if component == "vsh":
                filename1 = self.compose_filenames("mu", proc_number, 0)
                filename2 = self.compose_filenames("rhoinv", proc_number, 0)
                with open(filename1, "rb") as open_file:
                    field1 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                with open(filename2, "rb") as open_file:
                    field2 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                field = np.sqrt(field1 * field2)
            # - vsv
            if component == "vsv":
                filename1 = self.compose_filenames("mu", proc_number, 0)
                filename2 = self.compose_filenames("rhoinv", proc_number, 0)
                filename3 = self.compose_filenames("B", proc_number, 0)
                with open(filename1, "rb") as open_file:
                    field1 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                with open(filename2, "rb") as open_file:
                    field2 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                with open(filename3, "rb") as open_file:
                    field3 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                field = np.sqrt((field1 + field3) * field2)
        # - Reshape the array. ---------------------------------------------------------------------
        new_shape = [_i * _j for _i, _j in zip(shape[:3], shape[3:])]
        field = np.rollaxis(np.rollaxis(field, 3, 1), 3, self.setup["lpd"] + 1)
        field = field.reshape(new_shape, order="C")
        return field        
    # ==============================================================================================
    # - Plot slice at constant colatitude.
    # ==============================================================================================
    def plot_lat_slice(self, component, lat, valmin, valmax, iteration=0, verbose=True):
        # - Some initialisations. ------------------------------------------------------------------
        colat = np.pi * (90.-lat) / 180.0
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        vmax = float("-inf")
        vmin = float("inf")
        fig, ax = plt.subplots()
        # - Loop over processor boxes and check if colat falls within the volume. ------------------
        for p in range(n_procs):
            if (colat >= self.theta[p,:].min()) & (colat <= self.theta[p,:].max()):
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component,p,iteration)
                r, lon = np.meshgrid(self.z[p,:], self.phi[p,:])
                x = r * np.cos(lon)
                y = r * np.sin(lon)
                # - Find the colat index and plot for this one box. --------------------------------
                idx=min(np.where(min(np.abs(self.theta[p,:]-colat))==np.abs(self.theta[p,:]-colat))[0])
                colat_effective = self.theta[p,idx]*180.0/np.pi
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[idx,:,:].max())
                vmin = min(vmin, field[idx,:,:].min())
                # - Make a nice colourmap and plot. ------------------------------------------------
                my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],\
                    0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
                cax = ax.pcolor(x, y, field[idx,:,:], cmap=my_colormap, vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = fig.colorbar(cax)
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        plt.suptitle("Vertical slice of %s at %i degree colatitude" % (component, colat_effective), size="large")
        plt.axis('equal')
        plt.show()
        return
    
    def plot_lat_depth_lon_slice(self, component, lat, depth, minlon, maxlon, valmin, valmax, outfname=None, iteration=0, verbose=True):
        # - Some initialisations. ------------------------------------------------------------------
        colat = np.pi * (90.-lat) / 180.0
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        vmax = float("-inf")
        vmin = float("inf")
        radius = 1000.0 * (6371.0 - depth);
        print radius
        minlon=np.pi*minlon/180.0;
        maxlon=np.pi*maxlon/180.0;
        fig, ax = plt.subplots()
        # - Loop over processor boxes and check if colat falls within the volume. ------------------
        for p in range(n_procs):
            if (colat >= self.theta[p,:].min()) and (colat <= self.theta[p,:].max()) and (radius <= self.z[p,:].max())\
                and (minlon <= self.phi[p,:].min()) and (maxlon >= self.phi[p,:].max()):
                print 6371.-self.z[p,:].min()/1000.
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component,p,iteration)
                r, lon = np.meshgrid(self.z[p,:], self.phi[p,:])
                x = r * np.cos(lon)
                y = r * np.sin(lon)
                # - Find the colat index and plot for this one box. --------------------------------
                idx=min(np.where(min(np.abs(self.theta[p,:]-colat))==np.abs(self.theta[p,:]-colat))[0])
                colat_effective = self.theta[p,idx]*180.0/np.pi
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[idx,:,:].max())
                vmin = min(vmin, field[idx,:,:].min())
                # - Make a nice colourmap and plot. ------------------------------------------------
                my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],\
                    0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
                cax = ax.pcolor(x, y, field[idx,:,:], cmap=my_colormap, vmin=valmin,vmax=valmax)
                # if outfname !=None:
                    
        # - Add colobar and title. ------------------------------------------------------------------
        cb = fig.colorbar(cax)
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        plt.suptitle("Vertical slice of %s at %i degree colatitude" % (component, colat_effective), size="large")
        plt.axis('equal')
        plt.show()
        return
    # ==============================================================================================
    # - Plot depth slice.
    # ==============================================================================================
    def plot_depth_slice(self, component, depth, valmin, valmax, prefix='mywavefield',
        iteration=0, verbose=True, stations=True, res="i", mapflag='regional_ortho', mapfactor=2, geopolygons=[]):
        """
        plot_depth_slice(self, component, depth, valmin, valmax, iteration=0, verbose=True, stations=True, res="i")
        Plot depth slices of field component at depth "depth" with colourbar ranging between "valmin" and "valmax".
        The resolution of the coastline is "res" (c, l, i, h, f).
        The currently available "components" are:
            Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
            Velocity field snapshots: vx, vy, vz
            Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        mapflag - flag for map projection 
        """
        # - Some initialisations. ------------------------------------------------------------------
        fig=plt.figure()
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        radius = 1000.0 * (6371.0 - depth)
        vmax = float("-inf")
        vmin = float("inf")
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
        lat_centre = 90.0-lat_centre
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)
        
        # - Set up the map. ------------------------------------------------------------------------
        if mapflag=='global':
            m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif mapflag=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
            m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/mapfactor)
            # labels = [left,right,top,bottom]
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0))	
            # m.drawparallels(np.arange(-90.,120.,30.))
            # m.drawmeridians(np.arange(0.,360.,60.))
        elif mapflag=='regional_merc':
            m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
            m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
            m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        m.drawcoastlines()
        m.fillcontinents(lake_color='white',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        m.drawcountries()
        if len(geopolygons)!=0:
            geopolygons.PlotPolygon(mybasemap=m)

        # m.shadedrelief()
        # plt.show()
        # return
        # - Loop over processor boxes and check if depth falls within the volume. ------------------
        for p in range(n_procs):
            # print self.z[p,:].min(), self.z[p,:].max(), radius
            if (radius >= self.z[p,:].min()) & (radius <= self.z[p,:].max()):
                
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component,p,iteration)
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Find the depth index and plot for this one box. --------------------------------
                idz=min(np.where(min(np.abs(self.z[p,:]-radius))==np.abs(self.z[p,:]-radius))[0])
                r_effective = int(self.z[p,idz]/1000.0)
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[:,:,idz].max())
                vmin = min(vmin, field[:,:,idz].min())
                # - Make lats and lons. ------------------------------------------------------------
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Rotate if necessary. -----------------------------------------------------------
                if self.rotangle != 0.0:
                    lat_rot = np.zeros(np.shape(lon),dtype=float)
                    lon_rot = np.zeros(np.shape(lat),dtype=float)
                    for idlon in np.arange(len(lons)):
                        for idlat in np.arange(len(lats)):
                            lat_rot[idlat,idlon],lon_rot[idlat,idlon] = rotate_coordinates(self.n,-self.rotangle,90.0-lat[idlat,idlon],lon[idlat,idlon])
                            lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                    lon = lon_rot
                    lat = lat_rot
                # - Make a nice colourmap. ---------------------------------------------------------
                my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],\
                    0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
                x, y = m(lon, lat)
                # Nlon=int((lon_max-lon_min)/d_lon)+1
                # Nlat=int((lat_max-lat_min)/d_lat)+1
                # xout = np.linspace(x[0][0],x[0][-1], int(x[0].size/2))
                # yout = np.linspace(y[0][0],y[-1][0], int(np.flipud(y[:, 0]).size/2))
                # xout, yout = np.meshgrid(xout, yout)
                # INZValue=mpl_toolkits.basemap.interp( field[:,:,idz], x[0], np.flipud(y[:, 0]), xout, np.flipud(yout))
                # im = m.pcolormesh(xout, yout, INZValue, shading='gouraud', cmap=my_colormap, vmin=valmin,vmax=valmax)
                im = m.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap=my_colormap, vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = m.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        plt.suptitle("Depth slice of %s at %i km" % (component, r_effective), fontsize=20)
        # - Plot stations if available. ------------------------------------------------------------
        if (self.stations == True) & (stations==True):
            x,y = m(self.stlons,self.stlats)
            for n in range(self.n_stations):
                plt.text(x[n],y[n],self.stnames[n][:4])
                plt.plot(x[n],y[n],'ro')
        plt.show()
        if verbose == True:
            print "minimum value: "+str(vmin)+", maximum value: "+str(vmax)
            
        return; 
    
    def MakeAnimation(self, component, depth, valmin, valmax, outdir, prefix='mywavefield',iter0=100, iterf=17100, \
            dsnap=100, stations=False, res="i", mapflag='regional', dpi=300, mapfactor=2, geopolygons=[]):
        outdir=outdir+'_'+str(depth);
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
            vfname=self.directory+'/vx_0_'+str(int(iteration));
            if not os.path.isfile(vfname):
                raise NameError('Velocity Snapshot:'+vfname+' does not exist!')
        fig=plt.figure(num=None, figsize=(10, 10), dpi=dpi, facecolor='w', edgecolor='k')
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
        lat_centre = 90.0-lat_centre
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)
        if mapflag=='global':
            mymap=Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            mymap.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            mymap.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
        elif mapflag=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
            mymap = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/mapfactor)
            mymap.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            mymap.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
        elif mapflag=='regional_merc':
            mymap=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
            mymap.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
            mymap.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])	
        mymap.drawcoastlines()
        mymap.fillcontinents(lake_color='#99ffff',zorder=0.2)
        mymap.drawmapboundary(fill_color="white")
        mymap.drawcountries()
        if len(geopolygons)!=0:
            geopolygons.PlotPolygon(mybasemap=mymap);
        iterLst=np.arange((iterf-iter0)/dsnap)*dsnap+iter0;
        iterLst=iterLst.tolist();
        for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
            self.PlotSnapshot(mymap=mymap, component=component, depth=depth, valmin=valmin, valmax=valmax, iteration=iteration, stations=stations);
            outpsfname=outdir+'/'+prefix+'_%06d.png' %(iteration);
            fig.savefig(outpsfname, format='png', dpi=dpi)
        return 
    
    def PlotSnapshot(self, mymap, component, depth, valmin, valmax, iteration, stations):
        print 'Plotting Snapshot for:',iteration,' steps!'
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        radius = 1000.0 * (6371.0 - depth)
        vmax = float("-inf")
        vmin = float("inf")
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
        lat_centre = 90.0-lat_centre
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)
        # - Set up the map. ------------------------------------------------------------------------
        # fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
        # if mapflag=='global':
        #     # m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
        #     mymap.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
        #     mymap.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
        # else:
        #     mymap=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
        #     # m = Basemap(projection='lcc', llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max, \
        #     #     lon_0=lon_centre, lat_0=lat_centre, resolution=res)
        #     mymap.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
        #     mymap.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        # mymap.drawcoastlines()
        # mymap.fillcontinents("0.9", zorder=0)
        # mymap.drawmapboundary(fill_color="white")
        # mymap.drawcountries()
        # - Loop over processor boxes and check if depth falls within the volume. ------------------
        for p in range(n_procs):
            if (radius >= self.z[p,:].min()) & (radius <= self.z[p,:].max()):
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component,p,iteration)
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Find the depth index and plot for this one box. --------------------------------
                idz=min(np.where(min(np.abs(self.z[p,:]-radius))==np.abs(self.z[p,:]-radius))[0])
                r_effective = int(self.z[p,idz]/1000.0)
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[:,:,idz].max())
                vmin = min(vmin, field[:,:,idz].min())
                # - Make lats and lons. ------------------------------------------------------------
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Rotate if necessary. -----------------------------------------------------------
                if self.rotangle != 0.0:
                    lat_rot = np.zeros(np.shape(lon),dtype=float)
                    lon_rot = np.zeros(np.shape(lat),dtype=float)
                    for idlon in np.arange(len(lons)):
                        for idlat in np.arange(len(lats)):
                            lat_rot[idlat,idlon],lon_rot[idlat,idlon] = rotate_coordinates(self.n,-self.rotangle,90.0-lat[idlat,idlon],lon[idlat,idlon])
                            lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                    lon = lon_rot
                    lat = lat_rot
                # - Make a nice colourmap. ---------------------------------------------------------
                my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],\
                    0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
                x, y = mymap(lon, lat)
                im=mymap.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap=my_colormap, vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = mymap.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        # - Plot stations if available. ------------------------------------------------------------
        if (self.stations == True) & (stations==True):
            x,y = mymap(self.stlons,self.stlats)
            for n in range(self.n_stations):
                plt.text(x[n],y[n],self.stnames[n][:4])
                plt.plot(x[n],y[n],'ro')
        return;
    
    def MakeAnimationParallel(self, evlo, evla, component, depth, valmin, valmax, outdir, prefix='mywavefield',\
            iter0=100, iterf=17100, dsnap=100, stations=False, res="i", mapflag='regional_ortho', dpi=300, mapfactor=2, geopolygons=[]):
        
        outdir=outdir+'_'+str(depth);
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
            vfname=self.directory+'/vx_0_'+str(int(iteration));
            if not os.path.isfile(vfname):
                raise NameError('Velocity Snapshot:'+vfname+' does not exist!')
        # fig=plt.figure(num=None, figsize=(10, 10), dpi=dpi, facecolor='w', edgecolor='k')
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
        lat_centre = 90.0-lat_centre
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)  
        iterLst=np.arange((iterf-iter0)/dsnap)*dsnap+iter0;
        iterLst=iterLst.tolist();
        PLOTSNAP = partial(Iter2Snapshot, sfield=self, evlo=evlo, evla=evla, component=component, depth=depth,\
            valmin=valmin, valmax=valmax, stations=stations, prefix=prefix, mapflag=mapflag, \
            outdir=outdir, dpi=dpi, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,\
            lat_centre=lat_centre, lon_centre=lon_centre, d_lon=d_lon, d_lat=d_lat, res=res, \
            mapfactor=mapfactor, geopolygons=geopolygons);
        pool =mp.Pool()
        pool.map(PLOTSNAP, iterLst) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Making Snapshots for Animation  ( Parallel ) !'
        return
        
    def CheckVelocityLimit(self, component):
        # - Some initialisations. ------------------------------------------------------------------
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        vmax = float("-inf")
        vmin = float("inf")
        # - Loop over processor boxes and check if depth falls within the volume. ------------------
        for p in range(n_procs):
            # - Read this field and make lats & lons. ------------------------------------------
            field = self.read_single_box(component,p,0);
            # for idz in np.arange(40):
            # - Find min and max values. -------------------------------------------------------
            vmax = max(vmax, field[:,:,:].max())
            cmin=field[:,:,:].min()
            vmin = min(vmin, cmin)
            # print field[:,:,idz].shape
            # if cmin==0:
            #     print 'processor: ',p, 'idz', idz, 'index: ', np.count_nonzero((field[:,:,idz]==0)*np.ones(field[:,:,idz].shape));
            # - Make lats and lons. ------------------------------------------------------------
        # if verbose == True:
        print component+": minimum value: "+str(vmin)+", maximum value: "+str(vmax)
        return;
    
    
    def MakeAnimationVCrossSection(self, component, lat, depth, minlon, maxlon, valmin, valmax, outdir, prefix='mywavefield_VCS',\
            iter0=100, iterf=17100, dsnap=100, dpi=300):
        
        outdir=outdir+'_VCS'+str(lat);
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
            vfname=self.directory+'/vx_0_'+str(int(iteration));
            if not os.path.isfile(vfname):
                raise NameError('Velocity Snapshot:'+vfname+' does not exist!')
        
        for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
            self.PlotSnapshot(mymap=mymap, component=component, depth=depth, valmin=valmin, valmax=valmax, iteration=iteration, stations=stations);
            outpsfname=outdir+'/'+prefix+'_%06d.png' %(iteration);
            fig.savefig(outpsfname, format='png', dpi=dpi)
        
    
    def PlotSnapshot_VCrossSection(self, component, lat, depth, minlon, maxlon, valmin, valmax, outfname=None, iteration=0, verbose=True):
        # - Some initialisations. ------------------------------------------------------------------
        fig=plt.figure(num=None, figsize=(10, 10), dpi=dpi, facecolor='w', edgecolor='k')
        colat = np.pi * (90.-lat) / 180.0
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        vmax = float("-inf")
        vmin = float("inf")
        radius = 1000.0 * (6371.0 - depth);
        minlon=np.pi*minlon/180.0;
        maxlon=np.pi*maxlon/180.0;
        fig, ax = plt.subplots()
        # - Loop over processor boxes and check if colat falls within the volume. ------------------
        for p in range(n_procs):
            if (colat >= self.theta[p,:].min()) and (colat <= self.theta[p,:].max()) and (radius <= self.z[p,:].max())\
                and (minlon <= self.phi[p,:].min()) and (maxlon >= self.phi[p,:].max()):
                print 6371.-self.z[p,:].min()/1000.
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component,p,iteration)
                r, lon = np.meshgrid(self.z[p,:], self.phi[p,:])
                x = r * np.cos(lon)
                y = r * np.sin(lon)
                # - Find the colat index and plot for this one box. --------------------------------
                idx=min(np.where(min(np.abs(self.theta[p,:]-colat))==np.abs(self.theta[p,:]-colat))[0])
                colat_effective = self.theta[p,idx]*180.0/np.pi
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[idx,:,:].max())
                vmin = min(vmin, field[idx,:,:].min())
                # - Make a nice colourmap and plot. ------------------------------------------------
                my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],\
                    0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
                cax = ax.pcolor(x, y, field[idx,:,:], cmap=my_colormap, vmin=valmin,vmax=valmax)
                # if outfname !=None:
                    
        # - Add colobar and title. ------------------------------------------------------------------
        cb = fig.colorbar(cax)
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        plt.suptitle("Vertical slice of %s at %i degree colatitude" % (component, colat_effective), size="large")
        plt.axis('equal')
        plt.show()
        return
    
def Iter2Snapshot(iterN, sfield, evlo, evla, component, depth, valmin, valmax, stations, prefix, mapflag, outdir, dpi, \
        lat_min, lat_max, lon_min, lon_max, lat_centre, lon_centre, d_lon, d_lat, res, mapfactor, geopolygons):
    print 'Plotting Snapshot for:',iterN,' steps!'
    # fig=plt.figure();
    
    n_procs = sfield.setup["procs"]["px"] * sfield.setup["procs"]["py"] * sfield.setup["procs"]["pz"]
    radius = 1000.0 * (6371.0 - depth);
    vmax = float("-inf");
    vmin = float("inf");
    # fig=plt.figure(num=iterN, figsize=(10, 10), dpi=dpi, facecolor='w', edgecolor='k')
    # - Set up the map. ------------------------------------------------------------------------
    if mapflag=='global':
        mymap=Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
        mymap.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
        mymap.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
    elif mapflag=='regional_ortho':
        m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
        mymap = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
            llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
        # mymap.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
        # mymap.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[0,1,1,0])
        # mymap.drawparallels(np.arange(-80.0,80.0,10.0), dashes=[10000,1], labels=[1,0,0,0], linewidth=2.0,\
        #                     color=(0.5019607843137255, 0.5019607843137255, 0.5019607843137255),fontsize=20)
        # mymap.drawmeridians(np.arange(-170.0,170.0,10.0), dashes=[10000,1],linewidth=2.0,\
        #                     color=(0.5019607843137255, 0.5019607843137255, 0.5019607843137255))
    elif mapflag=='regional_merc':
        mymap=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
        mymap.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
        mymap.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])	
    mymap.drawcoastlines()
    mymap.fillcontinents(lake_color='#99ffff',zorder=0.2)
    mymap.drawmapboundary(fill_color="white")
    mymap.drawcountries()
    evx, evy=mymap(evlo, evla)
    mymap.plot(evx, evy, 'yo', markersize=2)
    # - Loop over processor boxes and check if depth falls within the volume. ------------------
    for p in range(n_procs):
        if (radius >= sfield.z[p,:].min()) & (radius <= sfield.z[p,:].max()):
            # - Read this field and make lats & lons. ------------------------------------------
            field = sfield.read_single_box(component,p,iterN)
            lats = 90.0 - sfield.theta[p,:] * 180.0 / np.pi
            lons = sfield.phi[p,:] * 180.0 / np.pi
            lon, lat = np.meshgrid(lons, lats)
            # - Find the depth index and plot for this one box. --------------------------------
            idz=min(np.where(min(np.abs(sfield.z[p,:]-radius))==np.abs(sfield.z[p,:]-radius))[0])
            r_effective = int(sfield.z[p,idz]/1000.0)
            # - Find min and max values. -------------------------------------------------------
            vmax = max(vmax, field[:,:,idz].max())
            vmin = min(vmin, field[:,:,idz].min())
            # - Make lats and lons. ------------------------------------------------------------
            lats = 90.0 - sfield.theta[p,:] * 180.0 / np.pi
            lons = sfield.phi[p,:] * 180.0 / np.pi
            lon, lat = np.meshgrid(lons, lats)
            # - Rotate if necessary. -----------------------------------------------------------
            if sfield.rotangle != 0.0:
                lat_rot = np.zeros(np.shape(lon),dtype=float)
                lon_rot = np.zeros(np.shape(lat),dtype=float)
                for idlon in np.arange(len(lons)):
                    for idlat in np.arange(len(lats)):
                        lat_rot[idlat,idlon],lon_rot[idlat,idlon] = rotate_coordinates(sfield.n,-sfield.rotangle,90.0-lat[idlat,idlon],lon[idlat,idlon])
                        lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                lon = lon_rot
                lat = lat_rot
            # - Make a nice colourmap. ---------------------------------------------------------
            my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],\
                0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            # my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.3,0.0],\
            #     0.39:[1.0,0.7,0.0], 0.5:[0.92,0.92,0.92], 0.61:[0.0,0.6,0.7], 0.7:[0.0,0.3,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            # my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.3,0.0],\
            #     0.35:[1.0,0.7,0.0], 0.5:[0.92,0.92,0.92], 0.65:[0.0,0.6,0.7], 0.7:[0.0,0.3,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            x, y = mymap(lon, lat)
            
            #-- Interpolating at the points in xi, yi
            # Nlon=int((lon_max-lon_min)/d_lon)+1
            # Nlat=int((lat_max-lat_min)/d_lat)+1
            # ZValue=field[:,:,idz];
            # Size=ZValue.size;
            # mywaveF = interpolate.interp2d(x, y, ZValue, kind='linear');
            # 
            # # ZValue=ZValue.reshape(Size);
            # xi = np.linspace(x.min(), x.max(), Nlon)
            # yi = np.linspace(y.min(), y.max(), Nlat)
            # zi = mywaveF(xi, yi)
            # xi, yi = np.meshgrid(xi, yi)
            # zi = griddata(x, y, ZValue, xi, yi)
            
            # im=mymap.pcolormesh(xi, yi, zi, shading='gouraud', cmap=my_colormap, vmin=valmin,vmax=valmax)
            im=mymap.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap=my_colormap, vmin=valmin,vmax=valmax)
    # - Add colobar and title. ------------------------------------------------------------------
    cb = mymap.colorbar(im, "right", size="3%", pad='2%')
    if len(geopolygons)!=0:
        geopolygons.PlotPolygon(mybasemap=mymap);
    if component in UNIT_DICT:
        cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
    # - Plot stations if available. ------------------------------------------------------------
    if (sfield.stations == True) & (stations==True):
        x,y = mymap(sfield.stlons,sfield.stlats)
        for n in range(sfield.n_stations):
            plt.text(x[n],y[n],sfield.stnames[n][:4])
            plt.plot(x[n],y[n],'ro')
    outpsfname=outdir+'/'+prefix+'_%06d.png' %(iterN);
    savefig(outpsfname, format='png', dpi=dpi)
    # outpsfname=outdir+'/'+prefix+'_%06d.pdf' %(iterN);
    # savefig(outpsfname, format='pdf', dpi=dpi)
    return;

    
class ses3dModelGen(object):
    def __init__(self, minlat, maxlat, minlon, maxlon, depth, dlat, dlon, dz, inflag='block'):
        
        if inflag=='block':
            minlat=minlat-dlat[0]/2.;
            maxlat=maxlat+dlat[0]/2.;
            minlon=minlon-dlon[0]/2.;
            maxlon=maxlon+dlon[0]/2.;
        if depth.size!=dlat.size or dlat.size != dlon.size or dlon.size != dz.size:
            self.numsub=1;
            self.depth=np.array([depth[0]]);
            self.dx=np.array([dlat[0]]);
            self.dy=np.array([dlon[0]]);
            self.dz=np.array([dz[0]]);
        else:
            self.numsub=depth.size;
            self.depth=depth;
            self.dx=dlat;
            self.dy=dlon;
            self.dz=dz;
        self.xmin=90.-maxlat;
        self.xmax=90.-minlat;
        self.ymin= minlon;
        self.ymax=maxlon;
    
    def GetBlockArrLst(self):
        radius=6371;
        self.xArrLst=[];
        self.yArrLst=[];
        self.xGArrLst=[];
        self.yGArrLst=[];
        self.zGArrLst=[];
        self.rGArrLst=[];
        self.zArrLst=[];
        self.rArrLst=[];
        dx=self.dx;
        dy=self.dy;
        dz=self.dz;
        mindepth=0;
        for numsub in np.arange(self.numsub):
            x0=self.xmin+dx[numsub]/2.;
            y0=self.ymin+dy[numsub]/2.;
            xG0=self.xmin;
            yG0=self.ymin;
            r0=(radius-self.depth[numsub])+dz[numsub]/2.;
            z0=self.depth[numsub]-dz[numsub]/2.;
            rG0=radius-self.depth[numsub];
            zG0=self.depth[numsub];
            nx=int((self.xmax-self.xmin)/dx[numsub]);
            ny=int((self.ymax-self.ymin)/dx[numsub]);
            nGx=int((self.xmax-self.xmin)/dx[numsub])+1;
            nGy=int((self.ymax-self.ymin)/dx[numsub])+1;
            nz=int((self.depth[numsub]-mindepth)/dz[numsub]);
            nGz=int((self.depth[numsub]-mindepth)/dz[numsub])+1;
            xArr=x0+np.arange(nx)*dx[numsub];
            yArr=y0+np.arange(ny)*dy[numsub];
            xGArr=xG0+np.arange(nGx)*dx[numsub];
            yGArr=yG0+np.arange(nGy)*dy[numsub];
            zArr=z0-np.arange(nz)*dz[numsub];
            rArr=r0+np.arange(nz)*dz[numsub];
            zGArr=zG0-np.arange(nGz)*dz[numsub];
            rGArr=rG0+np.arange(nGz)*dz[numsub];
            self.xArrLst.append(xArr);
            self.yArrLst.append(yArr);
            self.xGArrLst.append(xGArr);
            self.yGArrLst.append(yGArr);
            self.zArrLst.append(zArr);
            self.rArrLst.append(rArr);
            self.zGArrLst.append(zGArr);
            self.rGArrLst.append(rGArr);
            mindepth=self.depth[numsub];
        return;
    
    def generateBlockFile(self, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        bxfname=outdir+'/block_x';
        byfname=outdir+'/block_y';
        bzfname=outdir+'/block_z';
        outbxArr=np.array([]);
        outbyArr=np.array([]);
        outbzArr=np.array([]);
        outbxArr=np.append(outbxArr, self.numsub);
        outbyArr=np.append(outbyArr, self.numsub);
        outbzArr=np.append(outbzArr, self.numsub);
    
        for numsub in self.numsub-np.arange(self.numsub)-1:
            nGx=self.xGArrLst[numsub].size;
            nGy=self.yGArrLst[numsub].size;
            nGz=self.zGArrLst[numsub].size;
            outbxArr=np.append(outbxArr, nGx);
            outbyArr=np.append(outbyArr, nGy);
            outbzArr=np.append(outbzArr, nGz);
            outbxArr=np.append(outbxArr, self.xGArrLst[numsub]);
            outbyArr=np.append(outbyArr, self.yGArrLst[numsub]);
            outbzArr=np.append(outbzArr, self.rGArrLst[numsub]);
        np.savetxt(bxfname, outbxArr, fmt='%g');
        np.savetxt(byfname, outbyArr, fmt='%g');
        np.savetxt(bzfname, outbzArr, fmt='%g');
        return;
    
    def generateVsLimitedGeoMap(self, datadir, outdir, Vsmin, avgfname=None, dataprx='', datasfx='_mod'):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        if avgfname!=None:
            avgArr=np.loadtxt(avgfname);
            adepth=avgArr[:,0];
            arho=avgArr[:,1];
            aVp=avgArr[:,2];
            aVs=avgArr[:,3];
        depthInter=np.array([]);
        NZ=np.array([0],dtype=int);
        Lnz=0;
        for numsub in self.numsub-np.arange(self.numsub)-1: # maxdep ~ 0
            depthInter=np.append(depthInter, self.zArrLst[numsub]);
            NZ=np.append(NZ, NZ[Lnz]+self.zArrLst[numsub].size);
            Lnz=Lnz+1;
        
        depthInter=depthInter[::-1]; # 0 ~ maxdep
        if avgfname!=None:
            for colat in self.xArrLst[0]:
                for lon in self.yArrLst[0]:
                    lat=90.-colat;
                    infname=datadir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx;
                    if not os.path.isfile(infname):
                        continue;
                    inArr=np.loadtxt(infname);
                    depth=inArr[:,0];
                    Vs=inArr[:,1];
                    Vp=inArr[:,2];
                    Rho=inArr[:,3];
                    VpInter=np.interp(depthInter, depth, Vp);
                    VsInter=np.interp(depthInter, depth, Vs);
                    RhoInter=np.interp(depthInter, depth, Rho);
                    
                    aVpInter=np.interp(depthInter, adepth, aVp);
                    aVsInter=np.interp(depthInter, adepth, aVs);
                    aRhoInter=np.interp(depthInter, adepth, arho);
                    
                    if Vsmin!=999:
                        LArr=VsInter<Vsmin;
                        if LArr.size!=0:
                            UArr=VsInter>=Vsmin;
                            Vs1=Vsmin*LArr;
                            Vs2=VsInter*UArr;
                            VsInter=npr.evaluate('Vs1+Vs2');
                            # Brocher's relation
                            Vpmin=0.9409+2.0947*Vsmin-0.8206*Vsmin**2+0.2683*Vsmin**3-0.0251*Vsmin**4;
                            Vp1=Vpmin*LArr;
                            Vp2=VpInter*UArr;
                            VpInter=npr.evaluate('Vp1+Vp2');
                            
                            Rhomin=1.6612*Vpmin-0.4721*Vpmin**2+0.0671*Vpmin**3-0.0043*Vpmin**4+0.000106*Vpmin**5;
                            Rho1=Rhomin*LArr;
                            Rho2=RhoInter*UArr;
                            RhoInter=npr.evaluate('Rho1+Rho2');
                            
                    VpInter=npr.evaluate('VpInter-aVpInter');
                    VsInter=npr.evaluate('VsInter-aVsInter');
                    RhoInter=npr.evaluate('RhoInter-aRhoInter');
                    
                    outArr=np.append(depthInter, VsInter )
                    outArr=np.append(outArr, VpInter )
                    outArr=np.append(outArr, RhoInter )
                    outArr=outArr.reshape((4, depthInter.size))
                    outArr=outArr.T;
                    outfname=outdir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx;
                    np.savetxt(outfname, outArr, fmt='%g')
        else:
            for colat in self.xArrLst[0]:
                for lon in self.yArrLst[0]:
                    lat=90.-colat;
                    infname=datadir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx;
                    if not os.path.isfile(infname):
                        continue;
                    inArr=np.loadtxt(infname);
                    depth=inArr[:,0];
                    Vs=inArr[:,1];
                    Vp=inArr[:,2];
                    Rho=inArr[:,3];
                    VpInter=np.interp(depthInter, depth, Vp);
                    VsInter=np.interp(depthInter, depth, Vs);
                    RhoInter=np.interp(depthInter, depth, Rho);
                    
                    if Vsmin!=999:
                        LArr=VsInter<Vsmin;
                        if LArr.size!=0:
                            UArr=VsInter>=Vsmin;
                            Vs1=Vsmin*LArr;
                            Vs2=VsInter*UArr;
                            VsInter=npr.evaluate('Vs1+Vs2');
                            # Brocher's relation
                            Vpmin=0.9409+2.0947*Vsmin-0.8206*Vsmin**2+0.2683*Vsmin**3-0.0251*Vsmin**4;
                            Vp1=Vpmin*LArr;
                            Vp2=VpInter*UArr;
                            VpInter=npr.evaluate('Vp1+Vp2');
                            
                            Rhomin=1.6612*Vpmin-0.4721*Vpmin**2+0.0671*Vpmin**3-0.0043*Vpmin**4+0.000106*Vpmin**5;
                            Rho1=Rhomin*LArr;
                            Rho2=RhoInter*UArr;
                            RhoInter=npr.evaluate('Rho1+Rho2');
                    
                    outArr=np.append(depthInter, VsInter )
                    outArr=np.append(outArr, VpInter )
                    outArr=np.append(outArr, RhoInter )
                    outArr=outArr.reshape((4, depthInter.size))
                    outArr=outArr.T;
                    outfname=outdir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx;
                    print outfname
                    np.savetxt(outfname,outArr,fmt='%g')
        return;
    
    def generate3DModelFile(self, datadir, outdir, avgfname=None, dataprx='', datasfx='_mod', Vsmin=999):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        if avgfname!=None:
            avgArr=np.loadtxt(avgfname);
            adepth=avgArr[:,0];
            arho=avgArr[:,1];
            aVp=avgArr[:,2];
            aVs=avgArr[:,3];
        
        dVpfname=outdir+'/dvp';
        dRhofname=outdir+'/drho';
        dVsvfname=outdir+'/dvsv';
        dVshfname=outdir+'/dvsh';
        
        outdVpArr=np.array([]);
        outdRhoArr=np.array([]);
        outdVsvArr=np.array([]);
        outdVshArr=np.array([]);
        
        outdVpArr=np.append(outdVpArr, self.numsub);
        outdRhoArr=np.append(outdRhoArr, self.numsub);
        outdVsvArr=np.append(outdVsvArr, self.numsub);
        outdVshArr=np.append(outdVshArr, self.numsub);
        depthInter=np.array([]);
        NZ=np.array([0],dtype=int);
        Lnz=0;
        for numsub in self.numsub-np.arange(self.numsub)-1: # maxdep ~ 0
            depthInter=np.append(depthInter, self.zArrLst[numsub]);
            NZ=np.append(NZ, NZ[Lnz]+self.zArrLst[numsub].size);
            Lnz=Lnz+1;
        
        depthInter=depthInter[::-1]; # 0 ~ maxdep
        L=depthInter.size;
        VpArrLst=[];
        VsArrLst=[];
        RhoArrLst=[];
        
        # dd=depthInter[::-1];
        # Lnz=0;
        # for numsub in self.numsub-np.arange(self.numsub)-1: # maxdep ~ 0
        #     # print NZ[Lnz+1]
        #     print dd[NZ[Lnz]:NZ[Lnz+1]];
        #     Lnz=Lnz+1;
        # return;
        LH=0;
        dVpmin=999;
        dVsmin=999;
        drhomin=999;
        dVpmax=-999;
        dVsmax=-999;
        drhomax=-999;
        if avgfname!=None:
            for colat in self.xArrLst[0]:
                for lon in self.yArrLst[0]:
                    lat=90.-colat;
                    infname=datadir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx;
                    inArr=np.loadtxt(infname);
                    depth=inArr[:,0];
                    Vs=inArr[:,1];
                    Vp=inArr[:,2];
                    Rho=inArr[:,3];
                    VpInter=np.interp(depthInter, depth, Vp);
                    VsInter=np.interp(depthInter, depth, Vs);
                    RhoInter=np.interp(depthInter, depth, Rho);
                    
                    aVpInter=np.interp(depthInter, adepth, aVp);
                    aVsInter=np.interp(depthInter, adepth, aVs);
                    aRhoInter=np.interp(depthInter, adepth, arho);
                    
                    if Vsmin!=999:
                        LArr=VsInter<Vsmin;
                        if VsInter.min()<Vsmin:
                            UArr=VsInter>=Vsmin;
                            Vs1=Vsmin*LArr;
                            Vs2=VsInter*UArr;
                            VsInter=npr.evaluate('Vs1+Vs2');
                            # Brocher's relation
                            Vpmin=0.9409+2.0947*Vsmin-0.8206*Vsmin**2+0.2683*Vsmin**3-0.0251*Vsmin**4;
                            Vp1=Vpmin*LArr;
                            Vp2=VpInter*UArr;
                            VpInter=npr.evaluate('Vp1+Vp2');
                            
                            Rhomin=1.6612*Vpmin-0.4721*Vpmin**2+0.0671*Vpmin**3-0.0043*Vpmin**4+0.000106*Vpmin**5;
                            Rho1=Rhomin*LArr;
                            Rho2=RhoInter*UArr;
                            RhoInter=npr.evaluate('Rho1+Rho2');
                            
                    VpInter=npr.evaluate('VpInter-aVpInter');
                    VsInter=npr.evaluate('VsInter-aVsInter');
                    RhoInter=npr.evaluate('RhoInter-aRhoInter');
                    
                    VpArrLst.append(VpInter[::-1]);
                    VsArrLst.append(VsInter[::-1]);
                    RhoArrLst.append(RhoInter[::-1]); # maxdep ~ 0
                    LH=LH+1;
        else:
            for colat in self.xArrLst[0]:
                for lon in self.yArrLst[0]:
                    lat=90.-colat;
                    infname=datadir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx;
                    inArr=np.loadtxt(infname);
                    depth=inArr[:,0];
                    Vs=inArr[:,1];
                    Vp=inArr[:,2];
                    Rho=inArr[:,3];
                    VpInter=np.interp(depthInter, depth, Vp);
                    VsInter=np.interp(depthInter, depth, Vs);
                    RhoInter=np.interp(depthInter, depth, Rho);
                    
                    if Vsmin!=999:
                        LArr=VsInter<Vsmin;
                        if VsInter.min()<Vsmin:
                            print "Revaluing: "+str(lon)+" "+str(lat)+" "+str(VsInter.min());
                            
                            UArr=VsInter>=Vsmin;
                            Vs1=Vsmin*LArr;
                            Vs2=VsInter*UArr;
                            VsInter=npr.evaluate('Vs1+Vs2');
                            # Brocher's relation
                            Vpmin=0.9409+2.0947*Vsmin-0.8206*Vsmin**2+0.2683*Vsmin**3-0.0251*Vsmin**4;
                            Vp1=Vpmin*LArr;
                            Vp2=VpInter*UArr;
                            VpInter=npr.evaluate('Vp1+Vp2');
                            
                            Rhomin=1.6612*Vpmin-0.4721*Vpmin**2+0.0671*Vpmin**3-0.0043*Vpmin**4+0.000106*Vpmin**5;
                            Rho1=Rhomin*LArr;
                            Rho2=RhoInter*UArr;
                            RhoInter=npr.evaluate('Rho1+Rho2');
                    # dVpmin=min(dVpmin, VpInter.min());
                    # dVpmax=max(dVpmax, VpInter.max());
                    # dVsmin=min(dVsmin, VsInter.min());
                    # dVsmax=max(dVsmax, VsInter.max());
                    # drhomin=min(drhomin, RhoInter.min());
                    # drhomax=max(drhomax, RhoInter.max());
                    # 
                    VpArrLst.append(VpInter[::-1]);
                    VsArrLst.append(VsInter[::-1]);
                    RhoArrLst.append(RhoInter[::-1]); # maxdep ~ 0
                    
                    
                    # # outfname=outdir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx;
                    # # outArr=np.append(depthInter, VsInter);
                    # # outArr=np.append(outArr, VpInter);
                    # # outArr=np.append(outArr, RhoInter);
                    # # outArr=outArr.reshape((4,L));
                    # # outArr=outArr.T;
                    # # np.savetxt(outfname,outArr,fmt='%g'); 
                    
                    LH=LH+1;
        print LH
        # print 'Vp:', dVpmin, dVpmax, 'Vs:',dVsmin,dVsmax,'rho:',drhomin,drhomax;
        print 'End of Reading data!'
        # return;
        Lnz=0;
        for numsub in self.numsub-np.arange(self.numsub)-1:
            nx=self.xArrLst[numsub].size;
            ny=self.yArrLst[numsub].size;
            nz=self.zArrLst[numsub].size;
            nblock=nx*ny*nz;
            outdVpArr=np.append(outdVpArr, nblock);
            outdRhoArr=np.append(outdRhoArr, nblock);
            outdVsvArr=np.append(outdVsvArr, nblock);
            outdVshArr=np.append(outdVshArr, nblock);
            # Lh=0;
            for Lh in np.arange(LH):
            # for colat in self.xArrLst[numsub]:
            #     for lon in self.yArrLst[numsub]:
                    # dVp=VpArrLst[Lh];
                    # dVp=dVp[NZ[Lnz]:NZ[Lnz+1]];
                    # dVs=VsArrLst[Lh];
                    # dVsv=dVs[NZ[Lnz]:NZ[Lnz+1]];
                    # dVsh=dVsv;
                    # dRho=RhoArrLst[Lh];
                    # dRho=dRho[NZ[Lnz]:NZ[Lnz+1]];
                    outdVpArr=np.append(outdVpArr, VpArrLst[Lh][NZ[Lnz]:NZ[Lnz+1]]);
                    outdRhoArr=np.append(outdRhoArr, RhoArrLst[Lh][NZ[Lnz]:NZ[Lnz+1]]);
                    outdVsvArr=np.append(outdVsvArr, VsArrLst[Lh][NZ[Lnz]:NZ[Lnz+1]]);
                    outdVshArr=np.append(outdVshArr, VsArrLst[Lh][NZ[Lnz]:NZ[Lnz+1]]);
                    # Lh=Lh+1;
            Lnz=Lnz+1;
        print 'Saving data!'
        np.savetxt(dVpfname, outdVpArr, fmt='%s');
        np.savetxt(dRhofname, outdRhoArr, fmt='%s');
        np.savetxt(dVsvfname, outdVsvArr, fmt='%s');
        np.savetxt(dVshfname, outdVshArr, fmt='%s');
        return;
    
    def CheckInputModel(self,datadir, dataprx='', datasfx='_mod'):
        L=0
        Le=0
        for numsub in np.arange(self.numsub):
            xArr=self.xArrLst[numsub];
            yArr=self.yArrLst[numsub];
            for x in xArr:
                for y in yArr:
                    lat=90.-x;
                    lon=y;
                    # print lon, lat
                    infname=datadir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx;
                    Le=Le+1;
                    if not os.path.isfile(infname):
                        print dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx + ' NOT exists!' ;
                        L=L+1;
                        Le=Le-1;
        print 'Number of lacked-data grid points: ',L, Le
        return;
        

class Q_model(object):
    
    def __init__(self, QArr=np.array([50.0, 100.0, 500.0]), fmin=0.01, fmax=0.1, NumbRM=3):
        self.QArr=QArr;
        self.fmin=fmin;
        self.fmax=fmax;
        self.NumbRM=NumbRM;
        return;
    
    def Qcontinuous(self, tau_min=1.0e-3, tau_max=1.0e2, plotflag=True ):
        #------------------
        #- initializations 
        #------------------
        #- make logarithmic frequency axis
        f=np.logspace(np.log10(self.fmin),np.log10(self.fmax),100);
        w=2*np.pi*f;
        #- compute tau from target Q
        Q=self.QArr[0];
        tau=2/(np.pi*Q);
        #----------------------------------------------------
        #- computations for continuous absorption-band model 
        #----------------------------------------------------
        A=1+0.5*tau*np.log((1+w**2*tau_max**2)/(1+w**2*tau_min**2))
        B=tau*(np.arctan(w*tau_max)-np.arctan(w*tau_min))
        self.Q_continuous=A/B;
        self.v_continuous=np.sqrt(2*(A**2+B**2)/(A+np.sqrt(A**2+B**2)));
        if plotflag==True:
            plt.subplot(121)
            plt.semilogx(f,1./self.Q_continuous,'k')
            plt.xlabel('frequency [Hz]')
            plt.ylabel('1/Q')
            plt.title('absorption (1/Q)')
            
            plt.subplot(122)
            plt.semilogx(f,self.v_continuous,'k')
            plt.xlabel('frequency [Hz]')
            plt.ylabel('v')
            plt.title('phase velocity')
            
            plt.show()
        return;
    
    def Qdiscrete(self, max_it=30000, T_0=0.2, d=0.9998, f_ref=1.0/20.0, alpha=0.0):
        """
        Computation and visualisation of a discrete absorption-band model.
        For a given array of target Q values, the code determines the optimal relaxation
        times and weights using simulated annealing algorithmn. This is done within in specified
        frequency range.
        Input:
        max_it - number of iterations
        T_0    - the initial random step length
        d      - the temperature decrease (from one sample to the next by a factor of d)
        f_ref  - Reference frequency in Hz
        alpha  - exponent (alpha) for frequency-dependent Q, set to 0 for frequency-independent Q
        
        """
        #------------------
        #- initialisations 
        #------------------
        #- make logarithmic frequency axis
        f=np.logspace(np.log10(self.fmin),np.log10(self.fmax),100);
        w=2.0*np.pi*f;
        #- compute tau from target Q at reference frequency
        tau=1.0/self.QArr;
        #- compute target Q as a function of frequency
        Q_target=np.zeros([len(self.QArr), len(f)]);
        for n in np.arange(len(self.QArr)):
            Q_target[n,:]=self.QArr[n]*(f/f_ref)**alpha;
        #- compute initial relaxation times: logarithmically distributed
        tau_min=1.0/self.fmax;
        tau_max=1.0/self.fmin;
        tau_s=np.logspace(np.log10(tau_min),np.log10(tau_max),self.NumbRM)/(2*np.pi);
        #- make initial weights
        D=np.ones(self.NumbRM);
        #*********************************************************************
        # STAGE I
        # Compute relaxation times for constant Q values and weights all equal
        #*********************************************************************
        #- compute initial Q -------------------------------------------------
        chi=0.0;
        for n in np.arange(len(self.QArr)):
            A=1.0;
            B=0.0;
            for p in np.arange(self.NumbRM):
                A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2);
                B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2);
            Q=A/B;
            chi+=sum((Q-self.QArr[n])**2/self.QArr[n]**2);
        #--------------------------------
        #- search for optimal parameters 
        #--------------------------------
        #- random search for optimal parameters ------------------------------
        D_test=np.array(np.arange(self.NumbRM),dtype=float);
        tau_s_test=np.array(np.arange(self.NumbRM),dtype=float);
        T=T_0;
        for it in np.arange(max_it):
            #- compute perturbed parameters ----------------------------------
            tau_s_test=tau_s*(1.0+(0.5-rd.rand(self.NumbRM))*T);
            D_test=D*(1.0+(0.5-rd.rand(1))*T); 
            #- compute test Q ------------------------------------------------
            chi_test=0.0;
            for n in np.arange(len(self.QArr)):
                A=1.0;
                B=0.0;
                for p in np.arange(self.NumbRM):
                    A+=tau[n]*(D_test[p]*w**2*tau_s_test[p]**2)/(1.0+w**2*tau_s_test[p]**2);
                    B+=tau[n]*(D_test[p]*w*tau_s_test[p])/(1.0+w**2*tau_s_test[p]**2);
                Q_test=A/B;
                chi_test+=sum((Q_test-self.QArr[n])**2/self.QArr[n]**2);
            #- compute new temperature ----------------------------------------
            T=T*d;
            #- check if the tested parameters are better ----------------------
            if chi_test<chi:
                D[:]=D_test[:];          # equivalent to D=D_test.copy()
                tau_s[:]=tau_s_test[:];
                chi=chi_test;
        #**********************************************************************
        # STAGE II
        # Compute weights for frequency-dependent Q with relaxation times fixed
        #**********************************************************************
        #- compute initial Q --------------------------------------------------
        chi=0.0;
        for n in np.arange(len(self.QArr)):
            A=1.0;
            B=0.0;
            for p in np.arange(self.NumbRM):
                A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2);
                B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2);
            Q=A/B;
            chi+=sum((Q-Q_target[n,:])**2/self.QArr[n]**2);
        #- random search for optimal parameters -------------------------------
        T=T_0;
        for it in np.arange(max_it):
            #- compute perturbed parameters -----------------------------------
            D_test=D*(1.0+(0.5-rd.rand(self.NumbRM))*T);
            #- compute test Q -------------------------------------------------
            chi_test=0.0;
            for n in np.arange(len(self.QArr)):
                A=1.0
                B=0.0
                for p in np.arange(self.NumbRM):
                    A+=tau[n]*(D_test[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2);
                    B+=tau[n]*(D_test[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2);
                Q_test=A/B;
                chi_test+=sum((Q_test-Q_target[n,:])**2/self.QArr[n]**2);
            #- compute new temperature ------------------
            T=T*d
            #- check if the tested parameters are better 
            if chi_test<chi:
                D[:]=D_test[:];
                chi=chi_test;
                
        # print 'Cumulative rms error:  ', np.sqrt(chi/(len(Q)*len(self.QArr)))
        #************************************************
        # STAGE III
        # Compute partial derivatives dD[:]/dalpha
        #************************************************
        #- compute perturbed target Q as a function of frequency
        Q_target_pert=np.zeros([len(self.QArr),len(f)]);
        for n in range(len(self.QArr)):
            Q_target_pert[n,:]=self.QArr[n]*(f/f_ref)**(alpha+0.1);
        #- make initial weights
        D_pert=np.ones(self.NumbRM);
        D_pert[:]=D[:];
        #- compute initial Q ------------------------------------------------------------------------------
        chi=0.0
        for n in np.arange(len(self.QArr)):
            A=1.0;
            B=0.0;
            for p in np.arange(self.NumbRM):
                A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2);
                B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2);
            Q=A/B;
            chi+=sum((Q-Q_target_pert[n,:])**2/self.QArr[n]**2);
        #- random search for optimal parameters -----------------------------------------------------------
        T=T_0;
        for it in np.arange(max_it):
            #- compute perturbed parameters ---------------------------------------------------------------
            D_test_pert=D_pert*(1.0+(0.5-rd.rand(self.NumbRM))*T);
            #- compute test Q -----------------------------------------------------------------------------
            chi_test=0.0;
            for n in np.arange(len(self.QArr)):
                A=1.0;
                B=0.0;
                for p in np.arange(self.NumbRM):
                    A+=tau[n]*(D_test_pert[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2);
                    B+=tau[n]*(D_test_pert[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2);
                Q_test=A/B;
                chi_test+=sum((Q_test-Q_target_pert[n,:])**2/self.QArr[n]**2);
            #- compute new temperature --------------------------------------------------------------------
            T=T*d;
            #- check if the tested parameters are better --------------------------------------------------
            if chi_test<chi:
                D_pert[:]=D_test_pert[:];
                chi=chi_test;
        #********************
        # Output 
        #********************
        #------------------------------------
        #- sort weights and relaxation times 
        #------------------------------------
        decorated=[(tau_s[i], D[i]) for i in range(self.NumbRM)];
        decorated.sort();
        tau_s=[decorated[i][0] for i in range(self.NumbRM)];
        D=[decorated[i][1] for i in range(self.NumbRM)];
        #-------------------------------------
        #- print weights and relaxation times 
        #-------------------------------------
        print 'Weights: \t\t', D
        print 'Relaxation times: \t', tau_s
        print 'Partial derivatives: \t', (D_pert - D)/0.1
        # print 'Cumulative rms error:  ', np.sqrt(chi/(len(Q)*len(self.QArr)))
        self.D=D;
        self.tau_s=tau_s;
        self.D_pert=D_pert;
        self.chi=chi;
        self.Q_target=Q_target;
        return;
    
    def PlotQdiscrete( self, D=None, tau_s=None, f_ref=1.0/20.0, alpha=0.0 ):
        if D==None or tau_s==None or D.size!=self.NumbRM:
            D=self.D;
            tau_s=self.tau_s;
        chiTotal=0.0;
        #- make logarithmic frequency axis
        f=np.logspace(np.log10(self.fmin),np.log10(self.fmax),100);
        w=2.0*np.pi*f;
        #- compute tau from target Q at reference frequency
        tau=1.0/self.QArr;
        #- compute target Q as a function of frequency
        Q_target=np.zeros([len(self.QArr), len(f)]);
        for n in np.arange(len(self.QArr)):
            Q_target[n,:]=self.QArr[n]*(f/f_ref)**alpha;
        #- minimum and maximum frequencies for plotting in Hz
        f_min_plot=0.5*self.fmin;
        f_max_plot=2.0*self.fmax;
        f_plot=np.logspace(np.log10(f_min_plot),np.log10(f_max_plot),100);
        w_plot=2.0*np.pi*f_plot;
        #-----------------------------------------------------
        #- plot Q and phase velocity as function of frequency 
        #-----------------------------------------------------
        for n in np.arange(len(self.QArr)):
            #- compute optimal Q model for misfit calculations
            A=1.0;
            B=0.0;
            for p in np.arange(self.NumbRM):
                A+=tau[n]*(D[p]*w**2*tau_s[p]**2)/(1.0+w**2*tau_s[p]**2);
                B+=tau[n]*(D[p]*w*tau_s[p])/(1.0+w**2*tau_s[p]**2);
            Q=A/B;
            chi=np.sqrt(sum((Q-Q_target[n])**2/Q_target[n]**2)/len(Q));
            chiTotal+=(chi**2);
            print 'Individual rms error for Q_0='+str(self.QArr[n])+':  '+str(chi);
            #- compute optimal Q model for plotting
            A=1.0;
            B=0.0;
            for p in np.arange(self.NumbRM):
                A+=tau[n]*(D[p]*w_plot**2*tau_s[p]**2)/(1.0+w_plot**2*tau_s[p]**2);
                B+=tau[n]*(D[p]*w_plot*tau_s[p])/(1.0+w_plot**2*tau_s[p]**2);
            Q_plot=A/B;
            v_plot=np.sqrt(2*(A**2+B**2)/(A+np.sqrt(A**2+B**2)));
    
            # plt.subplot(121);
            plt.subplot(111);
            plt.semilogx([self.fmin,self.fmin],[0.9*self.QArr[n],1.1*self.QArr[n]],'r');
            plt.semilogx([self.fmax,self.fmax],[0.9*self.QArr[n],1.1*self.QArr[n]],'r');
            plt.semilogx(f,Q_target[n,:],'r',linewidth=3);
            plt.semilogx(f_plot,Q_plot,'k',linewidth=3);
            plt.xlim([f_min_plot,f_max_plot]);
            plt.xlabel('frequency [Hz]');
            plt.ylabel('Q');
            plt.title('quality factor Q');
        
            # plt.subplot(122);
            # plt.semilogx([self.fmin,self.fmin],[0.9,1.1],'r');
            # plt.semilogx([self.fmax,self.fmax],[0.9,1.1],'r');
            # plt.semilogx(f_plot,v_plot,'k',linewidth=2);
            # plt.xlim([f_min_plot,f_max_plot]);
            # plt.xlabel('frequency [Hz]');
            # plt.ylabel('v');
            # plt.title('phase velocity');
            plt.show();
        #------------------------------
        #- stress relaxation functions 
        #------------------------------
        dt=min(tau_s)/10.0;
        t=np.arange(0.0,max(tau_s),dt);
        for i in range(len(self.QArr)):
            c=np.ones(len(t));
            for n in range(self.NumbRM):
                c+=tau[i]*D[n]*np.exp(-t/tau_s[n]);
            plt.plot(t,c);
            plt.text(5.0*dt,np.max(c),r'$Q_0=$'+str(self.QArr[i]));
        plt.xlabel('time [s]');
        plt.ylabel('C(t)');
        plt.title('stress relaxation functions');  
        plt.show();
        print 'Cumulative rms error:  ', np.sqrt(chiTotal/(len(self.QArr)))
        return
    
    def WriteRelax(self, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        outfname_tau=outdir+'/LF_RELAX_tau';
        outfname_D=outdir+'/LF_RELAX_D';
        
        np.savetxt(outfname_tau, self.tau_s, fmt='%g');
        np.savetxt(outfname_D, self.D, fmt='%g');
        return;
        
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
    
    