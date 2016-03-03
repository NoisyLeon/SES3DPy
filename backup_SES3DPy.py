# -*- coding: utf-8 -*-
"""
A python module serve as interface for SES3D and AxiSEM

:Methods:
    Generate recfile_x from station.lst
    Convert output seismograms to SAC/miniseed binary files
    aftan analysis (compiled from aftan-1.1)
    SNR analysis based on aftan results
    C3(Correlation of coda of Cross-Correlation) computation
    Generate Predicted Phase Velocity Curves for an array
    Generate Input for Barmin's Surface Wave Tomography Code
    Automatic Receiver Function Analysis( Iterative Deconvolution and Harmonic Stripping )
    Eikonal Tomography
    Helmholtz Tomography (To be added soon)
    Bayesian Monte Carlo Inversion of Surface Wave and Receiver Function datasets (To be added soon)
    Stacking/Rotation for Cross-Correlation Results from SEED2CORpp
    
:Dependencies:
    numpy 1.9.1
    matplotlib 1.4.3
    numexpr 2.3.1
    ObsPy 0.10.2
    pyfftw3 0.2.1
    pyaftan( compiled from aftan 1.1 )
    GMT 5.x.x (For Eikonal/Helmholtz Tomography)
    CURefPy ( A submodule for noisepy, designed for automatic receiver function analysis, by Lili Feng)
    
:Modified and restructrued from original TOOLS directory in ses3d package, several new functions added. 
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
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


import obspy.sac.sacio as sacio
import obspy.taup.taup


import glob, os

from matplotlib.pyplot import cm
import matplotlib.pylab as plb
import copy
import scipy.signal
import numexpr as npr
from functools import partial
import multiprocessing as mp
import math
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


class StaInfo(object):
    """
    An object contains a station information several methods for station related analysis.
    -----------------------------------------------------------------------------------------------------
    General Parameters:
    stacode     - station name
    network     - network
    virtual_Net - virtula network name
    chan        - channels for analysis
    lon,lat     - position for station
    elevation   - elevation
    start_date  - start date of deployment of the station
    end_date    - end date of deployment of the station
    chan        - channel name
    ccflag      - cross-correlation flag, used to control staPair generation ( not necessary for cross-correlation)
    -----------------------------------------------------------------------------------------------------
    """
    def __init__(self, stacode=None, network='', virtual_Net=None, lat=None, lon=None, \
        elevation=None,start_date=None, end_date=None, ccflag=None, chan=[]):

        self.stacode=stacode
        self.network=network
        self.virtual_Net=virtual_Net
        self.lon=lon
        self.lat=lat
        self.elevation=elevation
        self.start_date=start_date
        self.end_date=end_date
        self.chan=[]

  
    # Member Functions for Receiver Function Analysis
    
    def setChan(self, chan):
        self.chan=copy.copy(chan)

    def appendChan(self,chan):
        self.chan.append(copy.copy(chan))

    def get_contents(self):
        if self.stacode==None:
            print 'StaInfo NOT Initialized yet!'
            return
        if self.network!='':
            print 'Network:%16s' %(self.network)
        print 'Station:%20s' %(self.stacode)
        print 'Longtitude:%17.3f' %(self.lon)
        print 'Latitude:  %17.3f' %(self.lat)

        return
    
    def GetPoslon(self):
        if self.lon<0:
            self.lon=self.lon+360.;
        return
    
    def GetNeglon(self):
        if self.lon>180.:
            self.lon=self.lon-360.;
        return
    
    def SES3D2SAC(self, datadir, outdir, scaling=1e9):
        ses3dST=ses3d_seismogram();
        staname=self.network+'.'+self.stacode;
        if ses3dST.read(datadir, staname, integrate=True):
            obsST=ses3dST.Convert2ObsStream(self.stacode, self.network, scaling=scaling);
            for tr in obsST:
                outfname=outdir+'/'+tr.id+'.SAC';
                tr.write(outfname, format='sac');
        return
    
    def AxiSEM2SAC(self, evla, evlo, evdp, dt, datadir, outdir, scaling=1e9):
        outTr=obspy.core.trace.Trace();
        infname=datadir+'/'+self.stacode+'_'+self.network+'_disp.dat'
        if not os.path.isfile(infname):
            return
        InArr=np.loadtxt(infname);
        if InArr.size==0:
            return
        # print infname
        Amp=InArr[:,1];
        outTr.data=Amp;
        outTr.stats.channel='BXZ';
        outTr.stats.delta=dt;
        outTr.stats.station=self.stacode; # tr.id will be automatically assigned once given stacode and network
        outTr.stats.network=self.network;
        stla=self.lat;
        stlo=self.lon;
        dist, az, baz=obsGeo.gps2DistAzimuth(evla, evlo, stla, stlo); # distance is in m
        dist=dist/1000.;
        outTr.stats['sac']={};
        outTr.stats.sac.dist=dist;
        outTr.stats.sac.az=az;  
        outTr.stats.sac.baz=baz;
        outTr.stats.sac.idep=6; # Displacement in nm
        outTr.stats.sac.evlo=evlo;
        outTr.stats.sac.evla=evla;
        outTr.stats.sac.stlo=stlo;
        outTr.stats.sac.stla=stla;
        outTr.stats.sac.evdp=evdp;
        outTr.stats.sac.kuser1='AxiSEM';
        outfname=outdir+'/'+self.stacode+'_'+self.network+'.SAC';
        outTr.write(outfname,format='sac');
        return
        
    
class StaLst(object):
    """
    An object contains a station list(a list of StaInfo object) information several methods for station list related analysis.
        stations: list of StaInfo
    """
    def __init__(self,stations=None):
        self.stations=[]
        if isinstance(stations, StaInfo):
            stations = [stations]
        if stations:
            self.stations.extend(stations)

    def __add__(self, other):
        """
        Add two StaLst with self += other.
        """
        if isinstance(other, StaInfo):
            other = StaLst([other])
        if not isinstance(other, StaLst):
            raise TypeError
        stations = self.stations + other.stations
        return self.__class__(stations=stations)

    def __len__(self):
        """
        Return the number of Traces in the StaLst object.
        """
        return len(self.stations)

    def __getitem__(self, index):
        """
        __getitem__ method of obspy.Stream objects.
        :return: Trace objects
        """
        if isinstance(index, slice):
            return self.__class__(stations=self.stations.__getitem__(index))
        else:
            return self.stations.__getitem__(index)

    def append(self, station):
        """
        Append a single StaInfo object to the current StaLst object.
        """
        if isinstance(station, StaInfo):
            self.stations.append(station)
        else:
            msg = 'Append only supports a single StaInfo object as an argument.'
            raise TypeError(msg)
        return self

    def ReadStaList(self, stafile):
        """
        Read Sation List from a txt file
        stacode longitude latidute network
        """
        f = open(stafile, 'r')
        Sta=[]
        for lines in f.readlines():
            lines=lines.split()
            stacode=lines[0]
            lon=float(lines[1])
            lat=float(lines[2])
            network=''
            ccflag=None
            if len(lines)==5:
                try:
                    ccflag=int(lines[3])
                    network=lines[4]
                except ValueError:
                    ccflag=int(lines[4])
                    network=lines[3]
            if len(lines)==4:
                try:
                    ccflag=int(lines[3])
                except ValueError:
                    network=lines[3]
            netsta=network+'.'+stacode
            if Sta.__contains__(netsta):
                index=Sta.index(netsta)
                if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                    raise ValueError('Incompatible Station Location:' + netsta+' in Station List!')
                else:
                    print 'Warning: Repeated Station:' +netsta+' in Station List!'
                    continue
            Sta.append(netsta)
            self.append(StaInfo (stacode=stacode, network=network, lon=lon, lat=lat, ccflag=ccflag ))
            f.close()
        return

    def MakeDirs(self, outdir, dirtout='COR'):
        """
        Create directories for the station list.
        directories format:
        outdir/dirtout/stacode
        """
        for station in self.stations:
            if dirtout=='':
                odir=outdir+'/'+station.stacode
            else:
                odir=outdir+'/'+dirtout+'/'+station.stacode
            if not os.path.isdir(odir):
                os.makedirs(odir)
        return

    def SES3D2SAC(self, datadir, outdir, scaling=1e9):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        for station in self.stations:
            station.SES3D2SAC( datadir=datadir, outdir=outdir, scaling=scaling);
        print 'End of Converting SES3D seismograms to SAC files !'
        return;
        
    def SES3D2SACParallel(self, datadir, outdir, scaling=1e9):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        Ses3D2SAC = partial(StationSES3D2SAC, datadir=datadir, outdir=outdir, scaling=scaling)
        pool =mp.Pool()
        pool.map(Ses3D2SAC, self.stations) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Converting SES3D seismograms to SAC files ( Parallel ) !'
        return;

    def AxiSEM2SAC(self, evla, evlo, evdp, dt, datadir, outdir, scaling=1e9):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        for station in self.stations:
            station.AxiSEM2SAC( evla=evla, evlo=evlo, evdp=evdp, dt=dt, datadir=datadir, outdir=outdir, scaling=scaling);
        print 'End of Converting AxiSEM seismograms to SAC files !'
        return;
    


def StationSES3D2SAC(STA, datadir, outdir, scaling=1e9):
    STA.SES3D2SAC( datadir=datadir, outdir=outdir, scaling=scaling);
    return;

class SES3DInputGenerator(InputFileGenerator):
    
    def StaLst2Generator(self, SList):
        L=0
        for station in SList:
            stacode=station.stacode
            net=station.network
            lon=station.lon;
            lat=station.lat;
            elevation=station.elevation
            if elevation==None:
                elevation=0.0;
            temp_sta={ "id": net+'.'+stacode, "latitude": lat, "longitude": lon, "elevation_in_m": elevation };
            self.add_stations(temp_sta);
            L=L+1;
        
        if (L>800):
            print 'ATTENTION: Number of receivers larger than default value!!! Check Carefully before running!';
        return;
        
    def AddExplosion(self, lon, lat, depth, mag, magtype='moment', des='NorthKorea', origintime=obspy.UTCDateTime()):
        if magtype=='moment':
            M0=10**(1.5*mag+9.1);
            Miso=M0*math.sqrt(2./3.);
        temp_event={"latitude": lat,"longitude": lon, "depth_in_km": depth,
            "origin_time": origintime, "description": des,\
            "m_rr": Miso, "m_tt": Miso, "m_pp": Miso,\
            "m_rt": 0.0, "m_rp": 0.0, "m_tp": 0.0};
        self.add_events(temp_event);
        return;
    
    def SetConfig(self, num_timpstep, dt, nx_global, ny_global, nz_global, px, py, pz, minlat, maxlat, minlon, maxlon, mindep, maxdep, \
            isdiss=True, QrelaxT=[1.7308, 14.3961, 22.9973], Qweight=[2.5100, 2.4354, 0.0879], SimType=0, OutFolder="../OUTPUT", ):
        
        print 'ATTENTION: Have You Updated the SOURCE/ses3d_conf.h and recompile the code???!!!';
        if SimType==0:
            self.config.simulation_type = "normal simulation";
        elif SimType==1:
            self.config.simulation_type = "adjoint forward";
        elif SimType==2:
            self.config.simulation_type = "adjoint reverse";
        # SES3D specific configuration  
        self.config.output_folder = OutFolder;
    
        # Time configuration.
        self.config.number_of_time_steps = num_timpstep;
        self.config.time_increment_in_s = dt;
        if (nx_global/px-int(nx_global/px))!=0.0 or (ny_global/py-int(ny_global/py))!=0.0 or (nz_global/pz-int(nz_global/pz))!=0.0:
            raise ValueError('nx_global/px, ny_global/py, nz_global/pz must ALL be integer!');
        if int(nx_global/px)!=22 or int(ny_global/py)!=27 or int(nz_global/pz)!=7:
            print 'ATTENTION: elements in x/y/z direction per processor is NOT default Value! Check Carefully before running!';
        totalP=px*py*pz;
        if totalP%12!=0:
            raise ValueError('total number of processor must be 12N !');
        print 'Number of Nodes needed at Janus is: %g' %(totalP/12);
        # SES3D specific discretization
        self.config.nx_global = nx_global;
        self.config.ny_global = ny_global;
        self.config.nz_global = nz_global;
        self.config.px = px;
        self.config.py = py;
        self.config.pz = pz;
        
        # Configure the mesh.
        self.config.mesh_min_latitude = minlat;
        self.config.mesh_max_latitude = maxlat;
        self.config.mesh_min_longitude = minlon; 
        self.config.mesh_max_longitude = maxlon;
        self.config.mesh_min_depth_in_km = mindep;
        self.config.mesh_max_depth_in_km = maxdep;
        self.CheckCFLCondition(minlat, maxlat, minlon, maxlon, mindep, maxdep, nx_global, ny_global, nz_global, dt);
        
        # # Define the rotation. Take care this is defined as the rotation of the
        # # mesh.  The data will be rotated in the opposite direction! The following
        # # example will rotate the mesh 5 degrees southwards around the x-axis. For
        # # a definition of the coordinate system refer to the rotations.py file. The
        # # rotation is entirely optional.
        # gen.config.rotation_angle_in_degree = 5.0
        # gen.config.rotation_axis = [1.0, 0.0, 0.0]
    
        # Define Q
        self.config.is_dissipative = isdiss;
        self.config.Q_model_relaxation_times = QrelaxT;
        self.config.Q_model_weights_of_relaxation_mechanisms = Qweight;
        
        return;
    
    def CheckCFLCondition(self, minlat, maxlat, minlon, maxlon, mindep, maxdep, nx_global, ny_global, nz_global, dt, C=0.3 ):
        if not os.path.isfile('./PREM.mod'):
            raise NameError('PREM Model File NOT exist!');
        InArr=np.loadtxt('./PREM.mod');
        depth=InArr[:,1];
        Vp=InArr[:,4];
        Vpmax=Vp[depth>maxdep][0];
        maxabslat=max(abs(minlat), abs(maxlat));
        dlat=(maxlat-minlat)/nx_global;
        dlon=(maxlon-minlon)/ny_global;
        dz=(maxdep-mindep)/nz_global;
        distEW, az, baz=obsGeo.gps2DistAzimuth(maxabslat, 45, maxabslat, 45+dlon); # distance is in m
        distEWmin=distEW/1000.*(6371.-maxdep)/6371.;
        distNS, az, baz=obsGeo.gps2DistAzimuth(maxabslat, 45, maxabslat+dlat, 45); # distance is in m
        distNSmin=distNS/1000.*(6371.-maxdep)/6371.;
        dzmin=dz;
        dtEW=C*distEWmin/Vpmax;
        dtNS=C*distNSmin/Vpmax;
        dtZ=C*dz/Vpmax;
        if dt > dtEW or dt > dtNS or dt > dtZ:
            raise ValueError('Time step violates Courant-Frieddrichs-Lewy Condition: ',dt, dtEW, dtNS, dtZ);
        else:
            print 'Time Step: ',dt, dtEW, dtNS, dtZ;
        return;
    
    def make_stf(self, dt=0.10, nt=5000, fmin=1.0/200.0, fmax=1.0/2.0, plotflag=False):
        """
        Generate a source time function for ses3d by applying a bandpass filter to a Heaviside function.
    
        make_stf(dt=0.13, nt=4000, fmin=1.0/100.0, fmax=1.0/8.0, plotflag=False)
    
        dt: Length of the time step. Must equal dt in the event_* file.
        nt: Number of time steps. Must equal to or greater than nt in the event_* file.
        fmin: Minimum frequency of the bandpass.
        fmax: Maximum frequency of the bandpass.
        """
        #- Make time axis and original Heaviside function. --------------------------------------------
        t = np.arange(0.0,float(nt+1)*dt,dt)
        h = np.ones(len(t))
        #- Apply filters. -----------------------------------------------------------------------------
        h = flt.highpass(h, fmin, 1.0/dt, 3, zerophase=False)
        h = flt.lowpass(h, fmax, 1.0/dt, 5, zerophase=False)
    
        #- Plot output. -------------------------------------------------------------------------------
        if plotflag == True:
            #- Time domain.
            plt.plot(t,h,'k')
            plt.xlim(0.0,float(nt)*dt)
            plt.xlabel('time [s]')
            plt.title('source time function (time domain)')
            plt.show()
            #- Frequency domain.
            hf = np.fft.fft(h)
            f = np.fft.fftfreq(len(hf), dt)
            plt.semilogx(f,np.abs(hf),'k')
            plt.plot([fmin,fmin],[0.0, np.max(np.abs(hf))],'r--')
            plt.text(1.1*fmin, 0.5*np.max(np.abs(hf)), 'fmin')
            plt.plot([fmax,fmax],[0.0, np.max(np.abs(hf))],'r--')
            plt.text(1.1*fmax, 0.5*np.max(np.abs(hf)), 'fmax')
            plt.xlim(0.1*fmin,10.0*fmax)
            plt.xlabel('frequency [Hz]')
            plt.title('source time function (frequency domain)')
            plt.show()
            
        self.config.source_time_function = h;
    
    def WriteSES3D(self, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        self.write(format="ses3d_4_1", output_dir=outdir);
        return

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
        print 'source: colatitude={} deg, longitude={} deg, depth={} m'.format(self.sx,self.sy,self.sz)
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
    
    def Convert2ObsStream(self, stacode, network, scaling=1e9):
        ST=obspy.core.Stream();
        stla=90.-self.rx;
        stlo=self.ry;
        evla=90.-self.sx;
        evlo=self.sy;
        evdp=self.sz/1000.;
        dist, az, baz=obsGeo.gps2DistAzimuth(evla, evlo, stla, stlo); # distance is in m
        dist=dist/1000.;
        # Initialization
        tr_x=obspy.core.trace.Trace();
        tr_y=obspy.core.trace.Trace();
        tr_z=obspy.core.trace.Trace();
        # N component
        tr_x.data=self.trace_x*(-1.0)*scaling;
        tr_x.stats.channel='BXN';
        tr_x.stats.delta=self.dt;
        tr_x.stats.station=stacode; # tr.id will be automatically assigned once given stacode and network
        tr_x.stats.network=network;
        
        tr_x.stats['sac']={};
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
        tr_z.data=self.trace_z*scaling;
        tr_z.stats.channel='BXZ';
        tr_z.stats.delta=self.dt;
        tr_z.stats.station=stacode; # tr.id will be automatically assigned once given stacode and network
        tr_z.stats.network=network;
        
        tr_z.stats['sac']={};
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
        tr_y.data=self.trace_y*scaling;
        tr_y.stats.channel='BXE';
        tr_y.stats.delta=self.dt;
        tr_y.stats.station=stacode; # tr.id will be automatically assigned once given stacode and network
        tr_y.stats.network=network;
        
        tr_y.stats['sac']={};
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
        # fid=open('./ses3d_data/rotation_parameters.txt','r')
        fid=open('/projects/life9360/software/ses3d_r07_b/SCENARIOS/NORTH_AMERICA/TOOLS/rotation_parameters.txt','r')
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
        fid_m=open(directory+filename,'w')
        if verbose==True:
            print 'write to file '+directory+filename
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
                        r=self.m[n].r[k]
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
    def plot_slice(self, depth, min_val_plot=None, max_val_plot=None, colormap='tomo', res='i', save_under=None, verbose=False):
        """ plot horizontal slices through an ses3d model
        plot_slice(self,depth,colormap='tomo',res='i',save_under=None,verbose=False)
        depth=depth in km of the slice
        colormap='tomo','mono'
        res=resolution of the map, admissible values are: c, l, i, h f
        save_under=save figure as *.png with the filename "save_under". Prevents plotting of the slice.
        """
        radius=6371.0-depth
        #- set up a map and colourmap -----------------------------------------
        if self.global_regional=='regional':
          m=Basemap(projection='merc',llcrnrlat=self.lat_min,urcrnrlat=self.lat_max,llcrnrlon=self.lon_min,urcrnrlon=self.lon_max,lat_ts=20,resolution=res)
          m.drawparallels(np.arange(self.lat_min,self.lat_max,self.d_lon),labels=[1,0,0,1])
          m.drawmeridians(np.arange(self.lon_min,self.lon_max,self.d_lat),labels=[1,0,0,1])
        elif self.global_regional=='global':
          m=Basemap(projection='ortho',lon_0=self.lon_centre,lat_0=self.lat_centre,resolution=res)
          m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
          m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries()
        m.drawmapboundary(fill_color=[1.0,1.0,1.0])
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
                x,y=m(self.m[k].lon_rot[0:nx-1,0:ny-1],self.m[k].lat_rot[0:nx-1,0:ny-1])
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
        #- loop over subvolumes to plot ---------------------------------------
        for k in np.arange(len(N_list)):
            im=m.pcolormesh(x_list[k],y_list[k],self.m[N_list[k]].v[:,:,idz_list[k]],shading='gouraud', cmap=my_colormap,vmin=min_val_plot,vmax=max_val_plot)
          #if colormap=='mono':
            #cs=m.contour(x_list[k],y_list[k],self.m[N_list[k]].v[:,:,idz_list[k]], colors='r',linewidths=1.0)
            #plt.clabel(cs,colors='r')
        #- make a colorbar and title ------------------------------------------
        m.colorbar(im,"right", size="3%", pad='2%')
        plt.title(str(depth)+' km')
        #- save image if wanted -----------------------------------------------
        if save_under is None:
            plt.show()
        else:
            plt.savefig(save_under+'.png', format='png', dpi=200)
            plt.close()
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
    "vz": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
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
#         ==============================================================================================
#         - Plot slice at constant colatitude.
#         ==============================================================================================
    def plot_colat_slice(self, component, colat, valmin, valmax, iteration=0, verbose=True):
        # - Some initialisations. ------------------------------------------------------------------
        colat = np.pi * colat / 180.0
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
                my_colormap=cm.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],\
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
#         ==============================================================================================
#         - Plot depth slice.
#         ==============================================================================================
    def plot_depth_slice(self, component, depth, valmin, valmax, iteration=0, verbose=True, stations=True, res="i", mapflag='regional'):
        """
        plot_depth_slice(self, component, depth, valmin, valmax, iteration=0, verbose=True, stations=True, res="i")
        Plot depth slices of field component at depth "depth" with colourbar ranging between "valmin" and "valmax".
        The resolution of the coastline is "res" (c, l, i, h, f).
        The currently available "components" are:
            Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
            Velocity field snapshots: vx, vy, vz
            Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        """
        # - Some initialisations. ------------------------------------------------------------------
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
        else:
            m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
            # m = Basemap(projection='lcc', llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max, \
            #     lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
            m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        m.drawcoastlines()
        m.fillcontinents("0.9", zorder=0)
        m.drawmapboundary(fill_color="white")
        m.drawcountries()
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
                x, y = m(lon, lat)
                im = m.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap=my_colormap, vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = m.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        plt.suptitle("Depth slice of %s at %i km" % (component, r_effective), size="large")
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
    
    # def MakeAnimation(self, component, depth, valmin, valmax, browseflag=False, iter0=100, iterf=18100, dsnap=1000, stations=True, res="i", mapflag='regional'):
    #     for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
    #         vfname=self.directory+'/vx_0_'+str(int(iteration));
    #         if not os.path.isfile(vfname):
    #             raise NameError('Velocity Snapshot:'+vfname+' does not exist!')
    #     fig = plt.figure()
    #     lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
    #     lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
    #     lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
    #     lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
    #     lat_centre = (lat_max+lat_min)/2.0
    #     lon_centre = (lon_max+lon_min)/2.0
    #     lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
    #     lat_centre = 90.0-lat_centre
    #     d_lon = np.round((lon_max-lon_min)/10.0)
    #     d_lat = np.round((lat_max-lat_min)/10.0)
    #     mymap = Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res);
    #     mymap.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
    #     mymap.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
    #     mymap.drawcoastlines()
    #     mymap.fillcontinents("0.9", zorder=0)
    #     mymap.drawmapboundary(fill_color="white")
    #     mymap.drawcountries()
    #     images=[];
    #     for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
    #         myimage=self.PlotSnapshot(mymap=mymap, component=component, depth=depth, valmin=valmin, valmax=valmax, iteration=iteration)
    #         images.append((myimage, ));
    #     plt.close('all')
    #     mymap = Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res);
    #     mymap.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
    #     mymap.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
    #     mymap.drawcoastlines()
    #     mymap.fillcontinents("0.9", zorder=0)
    #     mymap.drawmapboundary(fill_color="white")
    #     mymap.drawcountries()
    #     # ani = animation.FuncAnimation(fig, images)
    #     return mymap, fig, images
    #     
    # def PlotSnapshot(self, mymap, component, depth, valmin, valmax, iteration, mapflag='regional'):
    #     print 'Plotting Snapshot for:',iteration,' steps!'
    #     n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
    #     radius = 1000.0 * (6371.0 - depth)
    #     vmax = float("-inf")
    #     vmin = float("inf")
    #     lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
    #     lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
    #     lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
    #     lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
    #     lat_centre = (lat_max+lat_min)/2.0
    #     lon_centre = (lon_max+lon_min)/2.0
    #     lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
    #     lat_centre = 90.0-lat_centre
    #     d_lon = np.round((lon_max-lon_min)/10.0)
    #     d_lat = np.round((lat_max-lat_min)/10.0)
    #     # - Set up the map. ------------------------------------------------------------------------
    #     # if mapflag=='global':
    #     #     # m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
    #     #     mymap.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
    #     #     mymap.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
    #     # else:
    #     #     # m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
    #     #     # m = Basemap(projection='lcc', llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max, \
    #     #     #     lon_0=lon_centre, lat_0=lat_centre, resolution=res)
    #     #     mymap.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
    #     #     mymap.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
    #     # mymap.drawcoastlines()
    #     # mymap.fillcontinents("0.9", zorder=0)
    #     # mymap.drawmapboundary(fill_color="white")
    #     # mymap.drawcountries()
    #     # - Loop over processor boxes and check if depth falls within the volume. ------------------
    #     for p in range(n_procs):
    #         if (radius >= self.z[p,:].min()) & (radius <= self.z[p,:].max()):
    #             # - Read this field and make lats & lons. ------------------------------------------
    #             field = self.read_single_box(component,p,iteration)
    #             lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
    #             lons = self.phi[p,:] * 180.0 / np.pi
    #             lon, lat = np.meshgrid(lons, lats)
    #             # - Find the depth index and plot for this one box. --------------------------------
    #             idz=min(np.where(min(np.abs(self.z[p,:]-radius))==np.abs(self.z[p,:]-radius))[0])
    #             r_effective = int(self.z[p,idz]/1000.0)
    #             # - Find min and max values. -------------------------------------------------------
    #             vmax = max(vmax, field[:,:,idz].max())
    #             vmin = min(vmin, field[:,:,idz].min())
    #             # - Make lats and lons. ------------------------------------------------------------
    #             lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
    #             lons = self.phi[p,:] * 180.0 / np.pi
    #             lon, lat = np.meshgrid(lons, lats)
    #             # - Rotate if necessary. -----------------------------------------------------------
    #             if self.rotangle != 0.0:
    #                 lat_rot = np.zeros(np.shape(lon),dtype=float)
    #                 lon_rot = np.zeros(np.shape(lat),dtype=float)
    #                 for idlon in np.arange(len(lons)):
    #                     for idlat in np.arange(len(lats)):
    #                         lat_rot[idlat,idlon],lon_rot[idlat,idlon] = rotate_coordinates(self.n,-self.rotangle,90.0-lat[idlat,idlon],lon[idlat,idlon])
    #                         lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
    #                 lon = lon_rot
    #                 lat = lat_rot
    #             # - Make a nice colourmap. ---------------------------------------------------------
    #             my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],\
    #                 0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
    #             x, y = mymap(lon, lat)
    #             im = plt.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap=my_colormap, vmin=valmin,vmax=valmax)
    #     # - Add colobar and title. ------------------------------------------------------------------
    #     # cb = mymap.colorbar(im, "right", size="3%", pad='2%')
    #     # if component in UNIT_DICT:
    #     #     cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
    #     # - Plot stations if available. ------------------------------------------------------------
    #     # if (self.stations == True) & (stations==True):
    #     #     x,y = m(self.stlons,self.stlats)
    #     #     for n in range(self.n_stations):
    #     #         plt.text(x[n],y[n],self.stnames[n][:4])
    #     #         plt.plot(x[n],y[n],'ro')
    #     return im   


    def MakeAnimation(self, component, depth, valmin, valmax, outdir, \
        browseflag=False, iter0=100, iterf=9100, dsnap=500, stations=True, res="i", mapflag='regional'):
        for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
            vfname=self.directory+'/vx_0_'+str(int(iteration));
            if not os.path.isfile(vfname):
                raise NameError('Velocity Snapshot:'+vfname+' does not exist!')
        # fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
        # lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        # lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        # lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        # lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        # lat_centre = (lat_max+lat_min)/2.0
        # lon_centre = (lon_max+lon_min)/2.0
        # lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
        # lat_centre = 90.0-lat_centre
        # d_lon = np.round((lon_max-lon_min)/10.0)
        # d_lat = np.round((lat_max-lat_min)/10.0)
        # mymap = Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res);
        # mymap.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
        # mymap.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        # mymap.drawcoastlines()
        # mymap.fillcontinents("0.9", zorder=0)
        # mymap.drawmapboundary(fill_color="white")
        # mymap.drawcountries()
        for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
            self.PlotSnapshot(component, depth, valmin, valmax, iteration, stations, res);
        #     outpsfname=outdir+'/mywavefield_'+str(iteration)+'.ps';
        #     fig.savefig(outpsfname, format='ps')
        
        # frames=Snap_Input_Gen( mymap=mymap, component=component, depth=depth, valmin=valmin, valmax=valmax, iter0=iter0,\
        #     iterf=iterf, dsnap=dsnap, stations=stations, res=res, mapflag=mapflag )
        # # frames=np.arange(i)
        # ani = animation.FuncAnimation(fig=fig, func=self.PlotSnapshot, frames=frames, interval=10, repeat=True)
        # for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
        #     outpsfname=outdir+'/mywavefield_'+str(iteration)+'.ps';
        #     plt.savefig(outpsfname,format='ps')
        # plt.show()
        # outvideofname=outdir+'/'+'mywavefield.mp4';
        # ani.save(outvideofname);
        return 


# def animate(i):
#     lons, lats =  np.random.random_integers(, 130, 2)
#     x, y = map(lons, lats)
#     point.set_data(x, y)
#     return point,
#         
#     # initialization function: plot the background of each frame
#     def initSnapshot():
#         plt.suptitle("Depth slice of %s at %i km" % (component, r_effective), size="large")
#         return;
#     
#     def Snap_Input_Gen(self, mymap, component, depth, valmin, valmax, browseflag=False, iter0=100, iterf=15100, dsnap=1000, stations=True, res="i", mapflag='regional'):
# 
#         for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
#             yield mymap, component, depth, valmin, valmax, iteration, stations, res, mapflag;
#     
    def PlotSnapshot(self, component, depth, valmin, valmax, iteration, stations, res, mapflag='regional'):
        # mymap, component, depth, valmin, valmax, iteration, stations, res, mapflag=InputData[0], \
        #     InputData[1], InputData[2], InputData[3], InputData[4], InputData[5], InputData[6], InputData[7],InputData[8];
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
        fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
        if mapflag=='global':
            # m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            mymap.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            mymap.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
        else:
            mymap=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
            # m = Basemap(projection='lcc', llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max, \
            #     lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            mymap.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
            mymap.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        mymap.drawcoastlines()
        mymap.fillcontinents("0.9", zorder=0)
        mymap.drawmapboundary(fill_color="white")
        mymap.drawcountries()
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
                mymap.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap='seismic_r', vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        # cb = mymap.colorbar(im, "right", size="3%", pad='2%')
        # if component in UNIT_DICT:
        #     cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        # - Plot stations if available. ------------------------------------------------------------
        # if (self.stations == True) & (stations==True):
        #     x,y = m(self.stlons,self.stlats)
        #     for n in range(self.n_stations):
        #         plt.text(x[n],y[n],self.stnames[n][:4])
        #         plt.plot(x[n],y[n],'ro')
        outdir='/lustre/janus_scratch/life9360/snapshots'
        outfname=outdir+'/mywavefield_'+str(iteration)+'.png';
        fig.savefig(outfname, format='png')
        return;
#         
#     
def Snap_Input_Gen(mymap, component, depth, valmin, valmax, iter0=100, iterf=15100, dsnap=1000, stations=True, res="i", mapflag='regional'):
    for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
        yield mymap, component, depth, valmin, valmax, iteration, stations, res, mapflag;        

        
# def testF(InputData):
#     mymap, component, depth, valmin, valmax, iteration, stations, res, mapflag=InputData[0], \
#         InputData[1], InputData[2], InputData[3], InputData[4], InputData[5], InputData[6], InputData[7],InputData[8];
#     print 'Plotting Snapshot for:',iteration,' steps!'
#     return

        
    
    


        
        
    