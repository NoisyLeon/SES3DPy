
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
    
    