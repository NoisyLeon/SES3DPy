# -*- coding: utf-8 -*-
"""
A python module for ses3d data post-processing
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import numpy as np
import pyasdf 
import stations
import obspy
import pyaftan as ftan  # Comment this line if you do not have pyaftan
import numpy as np
import numexpr as npr
import glob, os
from matplotlib.pyplot import cm
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import copy
import scipy.signal
from functools import partial
import multiprocessing
import obspy.geodetics
import time
import shutil
from subprocess import call
import warnings
from mpl_toolkits.basemap import Basemap
from lasif import colors
from obspy.core.util import AttribDict


class ftanParam(object):
    """ An object to handle ftan output parameters
    ===========================================================================
    Basic FTAN parameters:
    nfout1_1 - output number of frequencies for arr1, (integer*4)
    arr1_1   - preliminary results.
                Description: real*8 arr1(8,n), n >= nfin)
                arr1_1[0,:] -  central periods, s
                arr1_1[1,:] -  observed periods, s
                arr1_1[2,:] -  group velocities, km/s
                arr1_1[3,:] -  phase velocities, km/s or phase if nphpr=0, rad
                arr1_1[4,:] -  amplitudes, Db
                arr1_1[5,:] -  discrimination function
                arr1_1[6,:] -  signal/noise ratio, Db
                arr1_1[7,:] -  maximum half width, s
                arr1_1[8,:] -  amplitudes
    arr2_1   - final results with jump detection
    nfout2_1 - output number of frequencies for arr2, (integer*4)
                Description: real*8 arr2(7,n), n >= nfin)
                If nfout2 == 0, no final result.
                arr2_1[0,:] -  central periods, s
                arr2_1[1,:] -  observed periods, s
                arr2_1[2,:] -  group velocities, km/sor phase if nphpr=0, rad
                arr2_1[3,:] -  phase velocities, km/s
                arr2_1[4,:] -  amplitudes, Db
                arr2_1[5,:] -  signal/noise ratio, Db
                arr2_1[6,:] -  maximum half width, s
                arr2_1[7,:] -  amplitudes
    tamp_1   -  time to the beginning of ampo table, s (real*8)
    nrow_1   -  number of rows in array ampo, (integer*4)
    ncol_1   -  number of columns in array ampo, (integer*4)
    amp_1    -  Ftan amplitude array, Db, (real*8)
    ierr_1   - completion status, =0 - O.K.,           (integer*4)
                                 =1 - some problems occures
                                 =2 - no final results
    ----------------------------------------------------------------------------
    Phase-Matched-Filtered FTAN parameters:
    nfout1_2 - output number of frequencies for arr1, (integer*4)
    arr1_2   - preliminary results.
                Description: real*8 arr1(8,n), n >= nfin)
                arr1_2[0,:] -  central periods, s (real*8)
                arr1_2[1,:] -  apparent periods, s (real*8)
                arr1_2[2,:] -  group velocities, km/s (real*8)
                arr1_2[3,:] -  phase velocities, km/s (real*8)
                arr1_2[4,:] -  amplitudes, Db (real*8)
                arr1_2[5,:] -  discrimination function, (real*8)
                arr1_2[6,:] -  signal/noise ratio, Db (real*8)
                arr1_2[7,:] -  maximum half width, s (real*8)
                arr1_2[8,:] -  amplitudes 
    arr2_2   - final results with jump detection
    nfout2_2 - output number of frequencies for arr2, (integer*4)
                Description: real*8 arr2(7,n), n >= nfin)
                If nfout2 == 0, no final results.
                arr2_2[0,:] -  central periods, s (real*8)
                arr2_2[1,:] -  apparent periods, s (real*8)
                arr2_2[2,:] -  group velocities, km/s (real*8)
                arr2_2[3,:] -  phase velocities, km/s (real*8)
                arr2_2[4,:] -  amplitudes, Db (real*8)
                arr2_2[5,:] -  signal/noise ratio, Db (real*8)
                arr2_2[6,:] -  maximum half width, s (real*8)
                arr2_2[7,:] -  amplitudes
    tamp_2   -  time to the beginning of ampo table, s (real*8)
    nrow_2   -  number of rows in array ampo, (integer*4)
    ncol_2   -  number of columns in array ampo, (integer*4)
    amp_2    -  Ftan amplitude array, Db, (real*8)
    ierr_2   - completion status, =0 - O.K.,           (integer*4)
                                =1 - some problems occures
                                =2 - no final results
    ===========================================================================
    """
    def __init__(self):
        # Parameters for first iteration
        self.nfout1_1=0
        self.arr1_1=np.array([])
        self.nfout2_1=0
        self.arr2_1=np.array([])
        self.tamp_1=0.
        self.nrow_1=0
        self.ncol_1=0
        self.ampo_1=np.array([],dtype='float32')
        self.ierr_1=0
        # Parameters for second iteration
        self.nfout1_2=0
        self.arr1_2=np.array([])
        self.nfout2_2=0
        self.arr2_2=np.array([])
        self.tamp_2=0.
        self.nrow_2=0
        self.ncol_2=0
        self.ampo_2=np.array([])
        self.ierr_2=0
        # Flag for existence of predicted phase dispersion curve
        self.preflag=False
        self.station_id=None

    def writeDISP(self, fnamePR):
        """
        Write FTAN parameters to DISP files given a prefix.
        fnamePR: file name prefix
        _1_DISP.0: arr1_1
        _1_DISP.1: arr2_1
        _2_DISP.0: arr1_2
        _2_DISP.1: arr2_2
        """
        if self.nfout1_1!=0:
            f10=fnamePR+'_1_DISP.0'
            Lf10=self.nfout1_1
            outArrf10=np.arange(Lf10)
            for i in np.arange(7):
                outArrf10=np.append(outArrf10, self.arr1_1[i,:Lf10])
            outArrf10=outArrf10.reshape((8,Lf10))
            outArrf10=outArrf10.T
            np.savetxt(f10, outArrf10, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf')
        if self.nfout2_1!=0:
            f11=fnamePR+'_1_DISP.1'
            Lf11=self.nfout2_1
            outArrf11=np.arange(Lf11)
            for i in np.arange(6):
                outArrf11=np.append(outArrf11, self.arr2_1[i,:Lf11])
            outArrf11=outArrf11.reshape((7,Lf11))
            outArrf11=outArrf11.T
            np.savetxt(f11, outArrf11, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf')
        if self.nfout1_2!=0:
            f20=fnamePR+'_2_DISP.0'
            Lf20=self.nfout1_2
            outArrf20=np.arange(Lf20)
            for i in np.arange(7):
                outArrf20=np.append(outArrf20, self.arr1_2[i,:Lf20])
            outArrf20=outArrf20.reshape((8,Lf20))
            outArrf20=outArrf20.T
            np.savetxt(f20, outArrf20, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf')
        if self.nfout2_2!=0:
            f21=fnamePR+'_2_DISP.1'
            Lf21=self.nfout2_2
            outArrf21=np.arange(Lf21)
            for i in np.arange(6):
                outArrf21=np.append(outArrf21, self.arr2_2[i,:Lf21])
            outArrf21=outArrf21.reshape((7,Lf21))
            outArrf21=outArrf21.T
            np.savetxt(f21, outArrf21, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf')
        return
    
    def writeDISPbinary(self, fnamePR):
        """
        Write FTAN parameters to DISP files given a prefix.
        fnamePR: file name prefix
        _1_DISP.0: arr1_1
        _1_DISP.1: arr2_1
        _2_DISP.0: arr1_2
        _2_DISP.1: arr2_2
        """
        f10=fnamePR+'_1_DISP.0'
        np.savez(f10, self.arr1_1, np.array([self.nfout1_1]) )
        f11=fnamePR+'_1_DISP.1'
        np.savez(f11, self.arr2_1, np.array([self.nfout2_1]) )
        f20=fnamePR+'_2_DISP.0'
        np.savez(f20, self.arr1_2, np.array([self.nfout1_2]) )
        f21=fnamePR+'_2_DISP.1'
        np.savez(f21, self.arr2_2, np.array([self.nfout2_2]) )
        return
    

    def FTANcomp(self, inftanparam, compflag=4):
        """
        Compare aftan results for two ftanParam objects.
        """
        fparam1=self
        fparam2=inftanparam
        if compflag==1:
            obper1=fparam1.arr1_1[1,:fparam1.nfout1_1]
            gvel1=fparam1.arr1_1[2,:fparam1.nfout1_1]
            phvel1=fparam1.arr1_1[3,:fparam1.nfout1_1]
            obper2=fparam2.arr1_1[1,:fparam2.nfout1_1]
            gvel2=fparam2.arr1_1[2,:fparam2.nfout1_1]
            phvel2=fparam2.arr1_1[3,:fparam2.nfout1_1]
        elif compflag==2:
            obper1=fparam1.arr2_1[1,:fparam1.nfout2_1]
            gvel1=fparam1.arr2_1[2,:fparam1.nfout2_1]
            phvel1=fparam1.arr2_1[3,:fparam1.nfout2_1]
            obper2=fparam2.arr2_1[1,:fparam2.nfout2_1]
            gvel2=fparam2.arr2_1[2,:fparam2.nfout2_1]
            phvel2=fparam2.arr2_1[3,:fparam2.nfout2_1]
        elif compflag==3:
            obper1=fparam1.arr1_2[1,:fparam1.nfout1_2]
            gvel1=fparam1.arr1_2[2,:fparam1.nfout1_2]
            phvel1=fparam1.arr1_2[3,:fparam1.nfout1_2]
            obper2=fparam2.arr1_2[1,:fparam2.nfout1_2]
            gvel2=fparam2.arr1_2[2,:fparam2.nfout1_2]
            phvel2=fparam2.arr1_2[3,:fparam2.nfout1_2]
        else:
            obper1=fparam1.arr2_2[1,:fparam1.nfout2_2]
            gvel1=fparam1.arr2_2[2,:fparam1.nfout2_2]
            phvel1=fparam1.arr2_2[3,:fparam1.nfout2_2]
            obper2=fparam2.arr2_2[1,:fparam2.nfout2_2]
            gvel2=fparam2.arr2_2[2,:fparam2.nfout2_2]
            phvel2=fparam2.arr2_2[3,:fparam2.nfout2_2]
        plb.figure()
        ax = plt.subplot()
        ax.plot(obper1, gvel1, '--k', lw=3) #
        ax.plot(obper2, gvel2, '-.b', lw=3)
        plt.xlabel('Period(s)')
        plt.ylabel('Velocity(km/s)')
        plt.title('Group Velocity Comparison')
        if (fparam1.preflag==True and fparam2.preflag==True):
            plb.figure()
            ax = plt.subplot()
            ax.plot(obper1, phvel1, '--k', lw=3) #
            ax.plot(obper2, phvel2, '-.b', lw=3)
            plt.xlabel('Period(s)')
            plt.ylabel('Velocity(km/s)')
            plt.title('Phase Velocity Comparison')
        return

class ses3dtrace(obspy.core.trace.Trace):
    """
    ses3dtrace:
    A derived class inherited from obspy.core.trace.Trace. This derived class have a variety of new member functions
    """
    def init_ftanParam(self):
        """
        Initialize ftan parameters
        """
        self.ftanparam=ftanParam()
    def init_snrParam(self):
        """
        Initialize SNR parameters
        """
        self.SNRParam=snrParam()
    def reverse(self):
        """
        Reverse the trace
        """
        self.data=self.data[::-1]
        return
    
    def aftan(self, pmf=True, piover4=-1.0, vmin=1.5, vmax=5.0, tmin=4.0, \
        tmax=30.0, tresh=20.0, ffact=1.0, taperl=1.0, snr=0.2, fmatch=1.0, phvelname='', predV=np.array([])):
        """ (Automatic Frequency-Time ANalysis) aftan analysis:
        ===========================================================================================================
        Input Parameters:
        pmf        - flag for Phase-Matched-Filtered output (default: True)
        piover4    - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
        vmin       - minimal group velocity, km/s
        vmax       - maximal group velocity, km/s
        tmin       - minimal period, s
        tmax       - maximal period, s
        tresh      - treshold for jump detection, usualy = 10, need modifications
        ffact      - factor to automatic filter parameter, usualy =1
        taperl     - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
        snr        - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
        fmatch     - factor to length of phase matching window
        phvelname  - predicted phase velocity file name
        predV      - predicted phase velocity curve, period = predV[:, 0],  Vph = predV[:, 1]
        
        Output:
        self.ftanparam, a object of ftanParam class, to store output aftan results
        ===========================================================================================================
        References:
        Levshin, A. L., and M. H. Ritzwoller. Automated detection, extraction, and measurement of regional surface waves.
             Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkh?user Basel, 2001. 1531-1545.
        Bensen, G. D., et al. Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements.
             Geophysical Journal International 169.3 (2007): 1239-1260.
        """
        try:
            self.ftanparam
        except:
            self.init_ftanParam()
        try:
            dist=self.stats.sac.dist
        except:
            dist, az, baz=obspy.geodetics.gps2dist_azimuth(self.stats.sac.evla, self.stats.sac.evlo,
                                self.stats.sac.stla, self.stats.sac.stlo) # distance is in m
            self.stats.sac.dist=dist/1000.
            dist=dist/1000.
        # if (phvelname==''):
        #     phvelname='./ak135.disp'
        nprpv = 0
        phprper=np.zeros(300)
        phprvel=np.zeros(300)
        if predV.size != 0:
            phprper=predV[:,0]
            phprvel=predV[:,1]
            nprpv = predV[:,0].size
            phprper=np.append( phprper, np.zeros(300-phprper.size) )
            phprvel=np.append( phprvel, np.zeros(300-phprvel.size) )
            self.ftanparam.preflag=True
        elif os.path.isfile(phvelname):
            # print 'Using prefile:',phvelname
            php=np.loadtxt(phvelname)
            phprper=php[:,0]
            phprvel=php[:,1]
            nprpv = php[:,0].size
            phprper=np.append( phprper, np.zeros(300-phprper.size) )
            phprvel=np.append( phprvel, np.zeros(300-phprvel.size) )
            self.ftanparam.preflag=True
        else:
            warnings.warn('No predicted dispersion curve for:'+self.stats.network+'.'+self.stats.station, UserWarning, stacklevel=1)
        nfin = 64
        npoints = 5  #  only 3 points in jump
        perc    = 50.0 # 50 % for output segment
        tempsac=self.copy()
        tb=self.stats.sac.b
        length=len(tempsac.data)
        if length>32768:
            warnings.warn('Length of seismogram is larger than 32768!', UserWarning, stacklevel=1)
            nsam=32768
            tempsac.data=tempsac.data[:nsam]
            tempsac.stats.e=(nsam-1)*tempsac.stats.delta+tb
            sig=tempsac.data
        else:
            sig=np.append(tempsac.data, np.zeros( float(32768-tempsac.data.size) ) )
            nsam=int( float (tempsac.stats.npts) )### for unknown reasons, this has to be done, nsam=int(tempsac.stats.npts)  won't work as an input for aftan
        dt=tempsac.stats.delta
        # Start to do aftan utilizing pyaftan
        self.ftanparam.nfout1_1,self.ftanparam.arr1_1,self.ftanparam.nfout2_1,self.ftanparam.arr2_1,self.ftanparam.tamp_1, \
                self.ftanparam.nrow_1,self.ftanparam.ncol_1,self.ftanparam.ampo_1, self.ftanparam.ierr_1= ftan.aftanpg(piover4, nsam, \
                    sig, tb, dt, dist, vmin, vmax, tmin, tmax, tresh, ffact, perc, npoints, taperl, nfin, snr, nprpv, phprper, phprvel)
        if pmf==True:
            if self.ftanparam.nfout2_1<3:
                return
            npred = self.ftanparam.nfout2_1
            tmin2 = self.ftanparam.arr2_1[1,0]
            tmax2 = self.ftanparam.arr2_1[1,self.ftanparam.nfout2_1-1]
            pred=np.zeros((2,300))
            pred[:,0:100]=self.ftanparam.arr2_1[1:3,:]
            pred=pred.T
            self.ftanparam.nfout1_2,self.ftanparam.arr1_2,self.ftanparam.nfout2_2,self.ftanparam.arr2_2,self.ftanparam.tamp_2, \
                    self.ftanparam.nrow_2,self.ftanparam.ncol_2,self.ftanparam.ampo_2, self.ftanparam.ierr_2 = ftan.aftanipg(piover4,nsam, \
                        sig,tb,dt,dist,vmin,vmax,tmin2,tmax2,tresh,ffact,perc,npoints,taperl,nfin,snr,fmatch,npred,pred,nprpv,phprper,phprvel)
        self.ftanparam.station_id=self.stats.network+'.'+self.stats.station 
        return
    
    def get_ms(self, Vgr=None, period=10., wfactor=2.):
        """Get surface wave magnitude according to Russell's formula
        """
        dist=self.stats.sac.dist
        Delta=obspy.geodetics.kilometer2degrees(dist)
        dt=self.stats.delta
        fcorner=0.6/period/np.sqrt(Delta)
        if Vgr==None:
            try:
                self.ftanparam
            except:
                warnings.warn('No aftan precalculated results: '+ self.stats.network+'.'+self.stats.station, UserWarning, stacklevel=1)
                self.aftan()
            obsTArr=self.ftanparam.arr2_2[1,:self.ftanparam.nfout2_2]
            VgrArr=self.ftanparam.arr2_2[2,:self.ftanparam.nfout2_2]
            AmpArr=self.ftanparam.arr2_2[7,:self.ftanparam.nfout2_2]
            Vgr=np.interp(period, obsTArr, VgrArr )
            Amp=np.interp(period, obsTArr, AmpArr )
        minT = max(0., dist/Vgr-wfactor*period)
        maxT = min(dist/Vgr+wfactor*period, self.stats.npts*dt)
        ntapb = int(period/dt)
        ntape = int(period/dt)
        nb = int(minT/dt)
        ne = int(maxT/dt)+1
        dataT=self.taper(nb, ne, ntapb, ntape)
        tempTr=self.copy()
        tempTr.data=dataT
        fmin=1./period-fcorner
        fmax=1./period+fcorner
        tempTr.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=3, zerophase=True)
        ab=(np.abs(tempTr.data)).max()
        Ms=np.log10(ab) + 0.5*np.log10( np.sin(Delta*np.pi/180.) ) + 0.0031*((20./period)**1.8)*Delta\
            - 0.66*np.log10(20./period)-np.log10(fcorner)-0.43
        if Vgr==None:
            return ab, Ms, Vgr, Amp
        else:
            return ab, Ms
    
    
    def taper(self, nb, ne, ntapb, ntape):
        omb = np.pi/ntapb
        ome = np.pi/ntape
        ncorr = ne+ntape
        npts=self.stats.npts
        if ncorr>npts:
            ncorr=npts
        dataTapered=np.append(self.data[:ncorr], np.zeros( npts-ncorr ) )
        ##################################
        #zerp padding and cosine tapering
        ##################################
        # left end of the signal
        if nb-ntapb-1 > 0:
            dataTapered[:nb-ntapb-1]=0.
        if nb>ntapb:
            k=np.arange(ntapb+1)+nb-ntapb
            rwinb=(np.cos(omb*(nb-k))+1.)/2.
            dataTapered[nb-ntapb-1:nb]=rwinb*dataTapered[nb-ntapb-1:nb]
            sums = 2.*np.sum(rwinb)
        else:
            k=np.arange(nb)
            rwinb=(np.cos(omb*(nb-k))+1.)/2.
            dataTapered[:nb]=rwinb*dataTapered[:nb]
            sums = 2.*np.sum(rwinb)
        # right end of the signal
        if ne+ntape<npts:
            k=np.arange(ntape+1)+ne
            rwine=(np.cos(ome*(ne-k))+1.)/2.
            dataTapered[ne-1:ne+ntape] = dataTapered[ne-1:ne+ntape]*rwine
        elif ne < npts:
            k=np.arange(npts-ne)+ne
            rwine=(np.cos(ome*(ne-k))+1.)/2.
            dataTapered[ne:] = dataTapered[ne:]*rwine
        sums = sums+ne-nb-1
        c=np.sum(dataTapered[:ncorr])
        c=-c/sums
        # detrend
        if nb>ntapb:
            dataTapered[nb-ntapb-1:nb]=rwinb*c+dataTapered[nb-ntapb-1:nb]
        if ne+ntape<npts:
            dataTapered[ne-1:ne+ntape] = dataTapered[ne-1:ne+ntape] + rwine*c
        elif ne < npts:
            dataTapered[ne:] = dataTapered[ne:] + rwine*c
        dataTapered[nb:ne]=dataTapered[nb:ne]+c
        return dataTapered
        
    def plotftan(self, plotflag=3, sacname=''):
        """
        Plot ftan diagram:
        This function plot ftan diagram.
        =================================================
        Input Parameters:
        plotflag -
            0: only Basic FTAN
            1: only Phase Matched Filtered FTAN
            2: both
            3: both in one figure
        sacname - sac file name than can be used as the title of the figure
        =================================================
        """
        try:
            fparam=self.ftanparam
            if fparam.nfout1_1==0:
                return "Error: No Basic FTAN parameters!"
            dt=self.stats.delta
            dist=self.stats.sac.dist
            if (plotflag!=1 and plotflag!=3):
                v1=dist/(fparam.tamp_1+np.arange(fparam.ncol_1)*dt)
                ampo_1=fparam.ampo_1[:fparam.ncol_1,:fparam.nrow_1]
                obper1_1=fparam.arr1_1[1,:fparam.nfout1_1]
                gvel1_1=fparam.arr1_1[2,:fparam.nfout1_1]
                phvel1_1=fparam.arr1_1[3,:fparam.nfout1_1]
                plb.figure()
                ax = plt.subplot()
                p=plt.pcolormesh(obper1_1, v1, ampo_1, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_1, gvel1_1, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_1, phvel1_1, '--w', lw=3) #

                if (fparam.nfout2_1!=0):
                    obper2_1=fparam.arr2_1[1,:fparam.nfout2_1]
                    gvel2_1=fparam.arr2_1[2,:fparam.nfout2_1]
                    phvel2_1=fparam.arr2_1[3,:fparam.nfout2_1]
                    ax.plot(obper2_1, gvel2_1, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_1, phvel2_1, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin1=obper1_1[0]
                Tmax1=obper1_1[fparam.nfout1_1-1]
                vmin1= v1[fparam.ncol_1-1]
                vmax1=v1[0]
                plt.axis([Tmin1, Tmax1, vmin1, vmax1])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('Basic FTAN Diagram '+sacname,fontsize=15)

            if fparam.nfout1_2==0 and plotflag!=0:
                return "Error: No PMF FTAN parameters!"
            if (plotflag!=0 and plotflag!=3):
                v2=dist/(fparam.tamp_2+np.arange(fparam.ncol_2)*dt)
                ampo_2=fparam.ampo_2[:fparam.ncol_2,:fparam.nrow_2]
                obper1_2=fparam.arr1_2[1,:fparam.nfout1_2]
                gvel1_2=fparam.arr1_2[2,:fparam.nfout1_2]
                phvel1_2=fparam.arr1_2[3,:fparam.nfout1_2]
                plb.figure()
                ax = plt.subplot()
                p=plt.pcolormesh(obper1_2, v2, ampo_2, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_2, gvel1_2, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_2, phvel1_2, '--w', lw=3) #

                if (fparam.nfout2_2!=0):
                    obper2_2=fparam.arr2_2[1,:fparam.nfout2_2]
                    gvel2_2=fparam.arr2_2[2,:fparam.nfout2_2]
                    phvel2_2=fparam.arr2_2[3,:fparam.nfout2_2]
                    ax.plot(obper2_2, gvel2_2, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_2, phvel2_2, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin2=obper1_2[0]
                Tmax2=obper1_2[fparam.nfout1_2-1]
                vmin2= v2[fparam.ncol_2-1]
                vmax2=v2[0]
                plt.axis([Tmin2, Tmax2, vmin2, vmax2])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                # plt.xlabel('Period(s)', fontsize=15)
                # plt.ylabel('Velocity(km/s)', fontsize=15)
                plt.title('PMF FTAN Diagram '+sacname,fontsize=15)
                # plt.title('FTAN Diagram '+sacname,fontsize=25)

            if ( plotflag==3 ):
                v1=dist/(fparam.tamp_1+np.arange(fparam.ncol_1)*dt)
                ampo_1=fparam.ampo_1[:fparam.ncol_1,:fparam.nrow_1]
                obper1_1=fparam.arr1_1[1,:fparam.nfout1_1]
                gvel1_1=fparam.arr1_1[2,:fparam.nfout1_1]
                phvel1_1=fparam.arr1_1[3,:fparam.nfout1_1]
                plb.figure(num=None, figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
                ax = plt.subplot(2,1,1)
                p=plt.pcolormesh(obper1_1, v1, ampo_1, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_1, gvel1_1, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_1, phvel1_1, '--w', lw=3) #
                if (fparam.nfout2_1!=0):
                    obper2_1=fparam.arr2_1[1,:fparam.nfout2_1]
                    gvel2_1=fparam.arr2_1[2,:fparam.nfout2_1]
                    phvel2_1=fparam.arr2_1[3,:fparam.nfout2_1]
                    ax.plot(obper2_1, gvel2_1, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_1, phvel2_1, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin1=obper1_1[0]
                Tmax1=obper1_1[fparam.nfout1_1-1]
                vmin1= v1[fparam.ncol_1-1]
                vmax1=v1[0]
                plt.axis([Tmin1, Tmax1, vmin1, vmax1])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('Basic FTAN Diagram '+sacname)

                v2=dist/(fparam.tamp_2+np.arange(fparam.ncol_2)*dt)
                ampo_2=fparam.ampo_2[:fparam.ncol_2,:fparam.nrow_2]
                obper1_2=fparam.arr1_2[1,:fparam.nfout1_2]
                gvel1_2=fparam.arr1_2[2,:fparam.nfout1_2]
                phvel1_2=fparam.arr1_2[3,:fparam.nfout1_2]

                ax = plt.subplot(2,1,2)
                p=plt.pcolormesh(obper1_2, v2, ampo_2, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_2, gvel1_2, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_2, phvel1_2, '--w', lw=3) #

                if (fparam.nfout2_2!=0):
                    obper2_2=fparam.arr2_2[1,:fparam.nfout2_2]
                    gvel2_2=fparam.arr2_2[2,:fparam.nfout2_2]
                    phvel2_2=fparam.arr2_2[3,:fparam.nfout2_2]
                    ax.plot(obper2_2, gvel2_2, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_2, phvel2_2, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin2=obper1_2[0]
                Tmax2=obper1_2[fparam.nfout1_2-1]
                vmin2= v2[fparam.ncol_2-1]
                vmax2=v2[0]
                plt.axis([Tmin2, Tmax2, vmin2, vmax2])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('PMF FTAN Diagram '+sacname)
        except AttributeError:
            print 'Error: FTAN Parameters are not available!'
        return
     

class InputFtanParam(object): ###
    """
    A subclass to store input parameters for aftan analysis and SNR Analysis
    ===============================================================================================================
    Parameters:
    pmf         - flag for Phase-Matched-Filtered output (default: Fasle)
    piover4     - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
    vmin        - minimal group velocity, km/s
    vmax        - maximal group velocity, km/s
    tmin        - minimal period, s
    tmax        - maximal period, s
    tresh       - treshold for jump detection, usualy = 10, need modifications
    ffact       - factor to automatic filter parameter, usualy =1
    taperl      - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
    snr         - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
    fmatch      - factor to length of phase matching window
    fhlen       - half length of Gaussian width
    dosnrflag   - whether to do SNR analysis or not
    predV       - predicted phase velocity curve, period = predV[:, 0],  Vph = predV[:, 1]
    ===============================================================================================================
    """
    def __init__(self):
        self.pmf=False
        self.piover4=-1.0
        self.vmin=1.5
        self.vmax=5.0
        self.tmin=4.0
        self.tmax=30.0
        self.tresh=20.0
        self.ffact=1.0
        self.taperl=1.0
        self.snr=0.2
        self.fmatch=1.0
        self.fhlen=0.008
        self.dosnrflag=False
        self.predV=np.array([])

class ses3d_seismogram(object):
    """An object to handle txt seismogram files from SES3D
    ===================================================================================
    Parameters:
    nt          - NPTS
    dt          - time interval
    time        - time array
    rx,ry,rz    - receiver location in x/y/z direction
    sx,sy,sz    - source location in x/y/z direction
    trace_x/y/z - three component trace data
    trace       - single component trace data
    integrate   - whether to integrate or not
    ===================================================================================
    """
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
        self.trace=np.array([])
        self.integrate=True
    
    
    def read_stream(self, directory, staname, integrate=True, fmin=0.01, fmax=1):
        """ read three component seismograms
        ===================================================================================
        Input parameters:
        directory   - directory where seismograms are located
        staname     - name of the station(network.stacode, e.g. SES.0S0)
        integrate   - integrate original velocity seismograms to displacement seismograms
        fmin, fmax  - minimal/maximal frequency for prefilter
        ===================================================================================
        """
        prefix=staname+'.___'
        self.integrate=integrate
        if not ( os.path.isfile(directory+'/'+prefix+'.x') and os.path.isfile(directory+'/'+prefix+'.y')\
                and os.path.isfile(directory+'/'+prefix+'.z') ):
            print 'No output for ',staname
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
        # print 'receiver: colatitude={} deg, longitude={} deg, depth={} m'.format(self.rx,self.ry,self.rz)
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
            xdata=float(fx.readline().strip())
            ydata=float(fy.readline().strip())
            zdata=float(fz.readline().strip())
            if np.isnan(xdata):
                xdata=0.0
            if np.isnan(ydata):
                ydata=0.0
            if np.isnan(zdata):
                zdata=0.0
            self.trace_x[k]=xdata
            self.trace_y[k]=ydata
            self.trace_z[k]=zdata
        self.time=np.linspace(0,self.nt*self.dt,self.nt)
        # integrate to displacement seismograms =========================
        if integrate==True:
            self.trace_x=np.cumsum(self.trace_x)*self.dt
            self.trace_y=np.cumsum(self.trace_y)*self.dt
            self.trace_z=np.cumsum(self.trace_z)*self.dt
            # self.bandpass(fmin=fmin,fmax=fmax)
        # close files ===================================================
        fx.close()
        fy.close()
        fz.close()
        return True
    
    def read_trace(self, directory, staname, channel, integrate=True):
        """ read one component seismograms
        ===================================================================================
        Input parameters:
        directory   - directory where seismograms are located
        staname     - name of the station(network.stacode, e.g. SES.0S0)
        channel     - channel to be read
        integrate   - integrate original velocity seismograms to displacement seismograms
        fmin, fmax  - minimal/maximal frequency for prefilter
        ===================================================================================
        """
        prefix=staname+'.___.'
        self.integrate=integrate
        txtfname = directory+'/'+prefix+channel
        if not txtfname:
            print 'No output for ',staname
            return False
        # open files ====================================================
        with open(txtfname,'r') as fid:
            fid.readline()
            self.nt=int(fid.readline().strip().split('=')[1])
            self.dt=float(fid.readline().strip().split('=')[1])
            fid.readline()
            line=fid.readline().strip().split('=')
            self.rx=float(line[1].split('y')[0])
            self.ry=float(line[2].split('z')[0])
            self.rz=float(line[3])
            # print 'receiver: colatitude={} deg, longitude={} deg, depth={} m'.format(self.rx,self.ry,self.rz)
            fid.readline()
            line=fid.readline().strip().split('=')
            self.sx=float(line[1].split('y')[0])
            self.sy=float(line[2].split('z')[0])
            self.sz=float(line[3])
            self.trace=np.empty(self.nt,dtype=np.float64)
            for k in range(self.nt):
                indata=float(fid.readline().strip())
                if np.isnan(indata):
                    indata=0.0
                self.trace[k]=indata
            self.time=np.linspace(0,self.nt*self.dt,self.nt)
        if integrate==True:
            self.trace=np.cumsum(self.trace)*self.dt
            # self.bandpass(fmin=fmin,fmax=fmax)
        return True

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
    
    def bandpass(self,fmin,fmax):
        """Apply a zero-phase bandpass to all traces. 
        """
        try:
            self.trace_x=flt.bandpass(self.trace_x,fmin,fmax,1.0/self.dt,corners=2,zerophase=True)
            self.trace_y=flt.bandpass(self.trace_y,fmin,fmax,1.0/self.dt,corners=2,zerophase=True)
            self.trace_z=flt.bandpass(self.trace_z,fmin,fmax,1.0/self.dt,corners=2,zerophase=True)
        except:
            self.trace=flt.bandpass(self.trace,fmin,fmax,1.0/self.dt,corners=2,zerophase=True)
        
    def get_obspy_trace(self, stacode, network, channel='BXZ', VminPadding=-1, scaling=1e9, Nsnap=float('inf')):
        """ get obspy trace from self.trace
        ===================================================================================
        Input parameters:
        VminPadding - minimal velocity for zero padding
        ===================================================================================
        """
        tr=obspy.core.Trace()
        tr.stats['sac']={}
        stla=90.-self.rx
        stlo=self.ry
        evla=90.-self.sx
        evlo=self.sy
        evdp=self.sz/1000.
        dist, az, baz=obspy.geodetics.base.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
        dist=dist/1000.
        Ntmax=-1
        L=self.trace.size
        if channel=='BXN':
            factor=-1.0
        else:
            factor=1.0
        #########
        # custom setting Vminpadding
        #########
        if dist < 3500:
            VminPadding = 2.0
        elif dist < 4000.:
            VminPadding = 2.5
        else:
            VminPadding = 2.7
        # 
        # ##########
        ######################
        # Zero Padding
        ######################
        if VminPadding>0:
            Ntmax=int(dist/VminPadding/self.dt)
            if Ntmax>Nsnap:
                Ntmax=-1
        padding=False
        if Ntmax>0 and L>Ntmax:
            max1=np.abs(self.trace[:Ntmax]).max()
            max2=np.abs(self.trace[Ntmax:]).max()
            if max2>0.1*max1:
                padding=True
        if padding:
            print 'Do padding for:', stlo, stla, dist, Ntmax, L
            tempdata=self.trace[:Ntmax]*scaling*factor
            tr.data=np.pad(tempdata,(0,L-Ntmax) ,mode='linear_ramp', end_values=(0))
            tr.stats.sac.kuser2='padded'
        else:
            tr.data=self.trace*scaling*factor
        tr.stats.channel=channel
        tr.stats.delta=self.dt
        tr.stats.station=stacode # tr.id will be automatically assigned once given stacode and network
        tr.stats.network=network
        tr.stats.sac.dist=dist
        tr.stats.sac.az=az  
        tr.stats.sac.baz=baz
        tr.stats.sac.idep=6 # Displacement in nm
        tr.stats.sac.evlo=evlo
        tr.stats.sac.evla=evla
        tr.stats.sac.stlo=stlo
        tr.stats.sac.stla=stla
        tr.stats.sac.evdp=evdp
        tr.stats.sac.kuser1='ses3d_4'
        return tr
    
    def get_obspy_stream(self, stacode, network, VminPadding=-1, scaling=1e9, Nsnap=float('inf')):
        ST=obspy.core.Stream()
        stla=90.-self.rx
        stlo=self.ry
        evla=90.-self.sx
        evlo=self.sy
        evdp=self.sz/1000.
        dist, az, baz=obspy.geodetics.base.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
        dist=dist/1000.
        Ntmax=-1
        L=self.trace_x.size
        #########
        # custom setting Vminpadding
        #########
        if dist < 3500:
            VminPadding = 2.0
        elif dist < 4000.:
            VminPadding = 2.5
        else:
            VminPadding = 2.7
        
        ##########
        if VminPadding>0:
            Ntmax=int(dist/VminPadding/self.dt)
            if Ntmax>Nsnap:
                Ntmax=-1
        padding=False
        if Ntmax>0 and L>Ntmax:
            max1=np.abs(self.trace_z[:Ntmax]).max()
            max2=np.abs(self.trace_z[Ntmax:]).max()
            if max2>0.1*max1:
                padding=True
        # Initialization
        tr_x=obspy.core.trace.Trace()
        tr_y=obspy.core.trace.Trace()
        tr_z=obspy.core.trace.Trace()
        # N component
        tr_x.stats['sac']={}
        if padding:
            print 'Do padding for:', stlo, stla, dist, Ntmax, L
            Xdata=self.trace_x[:Ntmax]*(-1.0)*scaling
            tr_x.data=np.pad(Xdata,(0,L-Ntmax),mode='linear_ramp',end_values=(0) )
            tr_x.stats.sac.kuser2='padded'
        else:
            tr_x.data=self.trace_x*(-1.0)*scaling
        tr_x.stats.channel='BXN'
        tr_x.stats.delta=self.dt
        tr_x.stats.station=stacode # tr.id will be automatically assigned once given stacode and network
        tr_x.stats.network=network
        tr_x.stats.sac.dist=dist
        tr_x.stats.sac.az=az  
        tr_x.stats.sac.baz=baz
        tr_x.stats.sac.idep=6 # Displacement in nm
        tr_x.stats.sac.evlo=evlo
        tr_x.stats.sac.evla=evla
        tr_x.stats.sac.stlo=stlo
        tr_x.stats.sac.stla=stla
        tr_x.stats.sac.evdp=evdp
        tr_x.stats.sac.kuser1='ses3d_4'
        # Z component
        tr_z.stats['sac']={}
        if padding==True:
            Zdata=self.trace_z[:Ntmax]*scaling
            tr_z.data=np.pad(Zdata,(0,L-Ntmax,),mode='linear_ramp',end_values=(0) )
            tr_z.stats.sac.kuser2='padded'
        else:
            tr_z.data=self.trace_z*scaling
        # tr_z.data=self.trace_z*scaling
        tr_z.stats.channel='BXZ'
        tr_z.stats.delta=self.dt
        tr_z.stats.station=stacode # tr.id will be automatically assigned once given stacode and network
        tr_z.stats.network=network
        tr_z.stats.sac.dist=dist
        tr_z.stats.sac.az=az  
        tr_z.stats.sac.baz=baz
        tr_z.stats.sac.idep=6 # Displacement in nm
        tr_z.stats.sac.evlo=evlo
        tr_z.stats.sac.evla=evla
        tr_z.stats.sac.stlo=stlo
        tr_z.stats.sac.stla=stla
        tr_z.stats.sac.evdp=evdp
        tr_z.stats.sac.kuser1='ses3d_4'
        # E component
        tr_y.stats['sac']={}
        if padding:
            Ydata=self.trace_y[:Ntmax]*scaling
            tr_y.data=np.pad(Ydata,(0,L-Ntmax,),mode='linear_ramp',end_values=(0) )
            tr_y.stats.sac.kuser2='padded'
        else:
            tr_y.data=self.trace_y*scaling
        # tr_y.data=self.trace_y*scaling
        tr_y.stats.channel='BXE'
        tr_y.stats.delta=self.dt
        tr_y.stats.station=stacode # tr.id will be automatically assigned once given stacode and network
        tr_y.stats.network=network
        tr_y.stats.sac.dist=dist
        tr_y.stats.sac.az=az  
        tr_y.stats.sac.baz=baz
        tr_y.stats.sac.idep=6 # Displacement in nm
        tr_y.stats.sac.evlo=evlo
        tr_y.stats.sac.evla=evla
        tr_y.stats.sac.stlo=stlo
        tr_y.stats.sac.stla=stla
        tr_y.stats.sac.evdp=evdp
        tr_y.stats.sac.kuser1='ses3d_4'
        ST.append(tr_x)
        ST.append(tr_y)
        ST.append(tr_z)
        return ST
    
class ses3dASDF(pyasdf.ASDFDataSet):
    
    def readsac(self, datadir, comptype='BXZ', verbose=False, stafile= None, minlon= None, dlon=None, Nlon= None, minlat=None, dlat=None,  Nlat=None):
        """ Read SAC files into ASDF dataset according to given station list
        =====================================================================
        Input Parameters:
        stafile   - station list file name
        datadir   - data directory
        comptype  - component type, can be e(East), n(North), u(UP), all
        Output:
        self.waveforms
        =====================================================================
        """
        if comptype == 'all':
            comptype=['BXE', 'BXN', 'BXZ']
        else:
            comptype=[comptype]
        if stafile != None or ( minlon != None and dlon != None and Nlon != None and minlat !=None and dlat != None and Nlat !=None):
            SLst=stations.StaLst()
            if stafile !=None:
                SLst.read(stafile=stafile)
            else:
                SLst.homo_stalst(minlat = minlat, Nlat = Nlat, minlon=minlon, Nlon = Nlon, dlat=dlat, dlon=dlon, net='EA')
            StaInv=SLst.GetInventory() 
            # self.add_stationxml(StaInv)
            print 'Start reading sac files!'
            for component in comptype:
                for sta in SLst.stations:
                    sacfname = datadir+'/'+sta.network+'.'+sta.stacode+'..'+component+".SAC"
                    # if not os.path.isfile(sacfname):
                    # # strlat='%g' %(sta.lat*100)
                    # # strlon='%g' %(sta.lon*100)
                    # # stacode=str(strlon)+'S'+str(strlat)
                    # # sacfname = datadir+'/'+sta.network+'.'+stacode+'..'+component+".SAC"
                    if verbose == True:
                        print 'Reading sac file:', sacfname
                    tr=obspy.read(sacfname)[0]
                    tr.stats.network = sta.network
                    tr.stats.station = sta.stacode
                    self.add_waveforms(tr, tag='ses3d_raw')
        else:
            SLst=stations.StaLst()
            for component in comptype:
                for sacfname in glob.glob(datadir+'/*..'+component+".SAC"):
                    stacode = sacfname.split('.')[1]
                    if verbose:
                        print 'Reading sac file:', sacfname
                    tr=obspy.read(sacfname)[0]
                    self.add_waveforms(tr, tag='ses3d_raw')
                    if component == comptype[0]:
                        SLst.append(stations.StaInfo(stacode=tr.stats.station, network=tr.stats.network, lon=tr.stats.sac.stlo, lat=tr.stats.sac.stla))                
            StaInv=SLst.GetInventory() 
            self.add_stationxml(StaInv)
        print 'End reading sac files!'
        return
    
    def readtxt(self, datadir, stafile, channel='all', verbose=True, VminPadding=2.7, factor=1):
        """ Read txt seismograms into ASDF dataset according to given station list
        ===================================================================================
        Input Parameters:
        stafile   - station list file name
        datadir   - data directory
        Output:
        self.waveforms
        ===================================================================================
        """
        SLst = stations.StaLst()
        SLst.read(stafile=stafile)
        StaInv = SLst.GetInventory() 
        self.add_stationxml(StaInv)
        print 'Start reading txt files!'
        # return
        dictchan={'BXZ':'z', 'BXE':'y', 'BXN':'x'}
        for sta in SLst.stations:
            ses3dST = ses3d_seismogram()
            if channel == 'all':
                if ses3dST.read_stream(directory=datadir, staname=sta.network+'.'+sta.stacode):
                    if verbose:
                        print 'Reading txt file: '+sta.network+'.'+sta.stacode
                    instream=ses3dST.get_obspy_stream(stacode=sta.stacode, network=sta.network, VminPadding=VminPadding)
                    if factor!=1:
                        instream.decimate(factor=factor)
                    self.add_waveforms( instream , tag='ses3d_raw')
            else:
                if ses3dST.read_trace(directory=datadir, staname=sta.network+'.'+sta.stacode, channel=dictchan[channel]):
                    if verbose:
                        print 'Reading txt file: '+sta.network+'.'+sta.stacode
                    inTr=ses3dST.get_obspy_trace(stacode=sta.stacode, network=sta.network,
                            channel=channel, VminPadding=VminPadding)
                    if factor!=1:
                        inTr.decimate(factor=factor)
                    self.add_waveforms( inTr,tag='ses3d_raw')
        print 'End reading txt files!'
        return
    
    def decimate(self, factor, verbose=True):
        for wid in self.waveforms.list():
            stream = self.waveforms[wid].ses3d_raw.copy()
            stream.decimate(factor=factor)
            del self.waveforms[wid].ses3d_raw
            if verbose:
                print 'Decimating: '+wid
            self.add_waveforms( stream, tag='ses3d_raw')
            
    def zero_padding(self, vminpadding, verbose=True):
        try:
            evlo=self.events.events[0].origins[0].longitude
            evla=self.events.events[0].origins[0].latitude
        except:
            raise AttributeError('No event specified to the datasets!')
        f=open('max.lst', 'w')
        for wid in self.waveforms.list():
            stream = self.waveforms[wid].ses3d_raw.copy()
            stla, elev, stlo=self.waveforms[wid].coordinates.values()
            dist, az, baz=obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
            distance = dist/1000.
            Ntmax = int(distance/vminpadding/stream[0].stats.delta)
            Zdata=stream[0].data[:Ntmax]
            stream[0].data=np.pad(Zdata, (0, stream[0].data.size-Ntmax), mode='linear_ramp',end_values=(0) )
            if verbose:
                print 'Zero padding: '+wid
            del self.waveforms[wid].ses3d_raw
            self.add_waveforms( stream, tag='ses3d_raw')
            f.writelines('%s %g %g\n' %(wid, abs(stream[0].data).max(), Ntmax))
        f.close()

    def get_wavefield(self, time, minlon, dlon, Nlon, minlat, dlat,  Nlat, net='SES', projection='regional_ortho'):
        wavefArr = np.zeros((Nlon, Nlat))
        lonArr = minlon + np.arange(Nlon)*dlon
        latArr = minlat + np.arange(Nlat)*dlat
        maxlat = minlat + (Nlat-1)*dlat
        maxlon = minlon + (Nlon-1)*dlon
        dt = self.waveforms[net+'.0S0'].ses3d_raw[0].stats.delta
        Nt = int(time/dt)+1
        for ilon in xrange(Nlon):
            for ilat in xrange(Nlat):
                lon = lonArr[ilon]
                lat = latArr[ilat]
                stacode=str(ilon)+'S'+str(ilat)
                print stacode
                wavefArr[ilon, ilat] = self.waveforms[net+'.'+stacode].ses3d_raw[0].data[Nt]
        lonM, latM = np.meshgrid(lonArr, latArr)       
        # - Some initialisations. ------------------------------------------------------------------
        fig=plt.figure()
        lat_centre = (minlat+maxlat)/2.0
        lon_centre = (minlon+maxlon)/2.0
        d_lon = np.round((maxlon-minlon)/10.0)
        d_lat = np.round((maxlat-minlat)/10.0)
        # - Set up the map. ------------------------------------------------------------------------
        if projection=='global':
            m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif projection=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m = Basemap(projection='ortho',lon_0=minlon,lat_0=minlat, resolution='i',\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/2., urcrnry=m1.urcrnry/2.)
            # labels = [left,right,top,bottom]
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0))	
            # m.drawparallels(np.arange(-90.,120.,30.))
            # m.drawmeridians(np.arange(0.,360.,60.))
        elif projection=='regional_merc':
            m=Basemap(projection='merc',llcrnrlat=minlat,urcrnrlat=maxlat,llcrnrlon=minlon,urcrnrlon=maxlon,lat_ts=20,resolution=res)
            m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
            m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        m.drawcoastlines()
        m.fillcontinents(lake_color='white',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        m.drawcountries()
        # if len(geopolygons)!=0:
        #     geopolygons.PlotPolygon(mybasemap=m)
        cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
        x, y = m(lonM, latM)
        vlimit = max(wavefArr.max() ,abs(wavefArr.min())) / 1.
        # print x.shape, y.shape, wavefArr.shape
        im = m.pcolormesh(x, y, wavefArr.T, cmap='seismic_r', vmin=-vlimit,vmax=vlimit)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = m.colorbar(im, "right", size="3%", pad='2%')
        # if component in UNIT_DICT:
        #     cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        # plt.suptitle("Depth slice of %s at %i km" % (component, r_effective), fontsize=20)
        # - Plot stations if available. ------------------------------------------------------------
        # if (self.stations == True) & (stations==True):
        #     x,y = m(self.stlons,self.stlats)
        #     for n in range(self.n_stations):
        #         plt.text(x[n],y[n],self.stnames[n][:4])
        #         plt.plot(x[n],y[n],'ro')
        plt.show()
        return 
    
    def AddEvent(self, evlo, evla, evdp):
        """ Add event information to ASDF dataset
        =============================================================
        Input Parameters:
        evlo, evla, evdp      - event location, unit is km
        Output:
        self.events
        =============================================================
        """
        print 'Attention: Event Location unit is km!'
        origin=obspy.core.event.origin.Origin(longitude=evlo, latitude=evla, depth=evdp)
        event=obspy.core.event.event.Event(origins=[origin])
        # evname='E%03d' %self.events.count()
        # origin.comments.append(evname)
        # print evname
        catalog=obspy.core.event.catalog.Catalog(events=[event])
        self.add_quakeml(catalog)
        return
    
    def pre_php(self, outdir, mapfile='./MAPS/smpkolya_phv'):
        """
        Generate predicted phase velocity dispersion curves for all events.
        ======================================================================
        Input Parameters:
        outdir  - output directory
        mapfile - phase velocity maps
        ----------------------------------------------------------------------
        Input format:
        prephaseEXE pathfname mapfile perlst staname
        
        Output format:
        outdirL(outdirR)/sta1.sta2.pre
        ======================================================================
        """
        for i in xrange(self.events.count()):
            evid = 'E%03d' %i
            evlo = self.events[i].origins[0].longitude
            evla = self.events[i].origins[0].latitude
            pathfname=evid+'_pathfile'
            prephaseEXE='./mhr_grvel_predict/lf_mhr_predict_earth'
            perlst='./mhr_grvel_predict/perlist_phase'
            if not os.path.isfile(prephaseEXE):
                print 'lf_mhr_predict_earth executable does not exist!'
                return
            if not os.path.isfile(perlst):
                print 'period list does not exist!'
                return
            with open(pathfname,'w') as f:
                ista=0
                for station_id in self.waveforms.list():
                    stlo=self.waveforms[station_id].coordinates['longitude']
                    stla=self.waveforms[station_id].coordinates['latitude']
                    # print station_id
                    stacode=station_id.split('.')[1]
                    # if ( abs(stlo-evlo) < 0.1 and abs(stla-evla)<0.1 ):
                    #     continue
                    ista=ista+1
                    f.writelines('%5d%5d %15s %15s %10.5f %10.5f %10.5f %10.5f \n'
                            %(1, ista, evid, stacode, evla, evlo, stla, stlo ))
                    # f.writelines('%5d%5d %15s %15s %10.5f %10.5f %10.5f %10.5f \n'
                    #              %(1, ista, self.stacode, station.stacode, self.lat, self.lon, station.lat, station.lon ))
            call([prephaseEXE, pathfname, mapfile, perlst, evid])
            # os.remove(pathfname)
            outdirL=outdir+'_L'
            outdirR=outdir+'_R'
            if not os.path.isdir(outdirL):
                os.makedirs(outdirL)
            if not os.path.isdir(outdirR):
                os.makedirs(outdirR)
            fout = open(evid+'_temp','wb')
            for l1 in open('PREDICTION_L'+'_'+evid):
                l2 = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[3],l2[4])
                    fout = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[2],l2[3])
                    fout = open(outname,"w")                
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            for l1 in open('PREDICTION_R'+'_'+evid):
                l2 = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[3],l2[4])
                    fout = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[2],l2[3])
                    fout = open(outname,"w")         
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            fout.close()
            os.remove(evid+'_temp')
            os.remove('PREDICTION_L'+'_'+evid)
            os.remove('PREDICTION_R'+'_'+evid)
        return

    def aftan(self, channel='BXZ', tb=-13.5, outdir=None, inftan=InputFtanParam(), basic1=True, basic2=False,
            pmf1=False, pmf2=False, verbose=True, prephdir=None):
        """ aftan analysis for ASDF Dataset
        =======================================================================================
        Input Parameters:
        compindex   - component index in waveforms path (default = 0)
        tb          - begin time (default = -13.5)
        outdir      - directory for output disp txt files (default = None, no txt output)
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        Output:
        self.auxiliary_data.DISPbasic1, self.auxiliary_data.DISPbasic2,
        self.auxiliary_data.DISPpmf1, self.auxiliary_data.DISPpmf2
        =======================================================================================
        """
        print 'Start aftan analysis!'
        try:
            evlo=self.events.events[0].origins[0].longitude
            evla=self.events.events[0].origins[0].latitude
        except:
            raise AttributeError('No event specified to the datasets!')
        for station_id in self.waveforms.list():
            # Get data from ASDF dataset
            subdset = self.waveforms[station_id]
            tr=subdset.ses3d_raw.select(channel=channel)[0]
            # # # tr=self.waveforms[station_id].ses3d_raw.select(channel=channel)[0]
            tr.stats.sac={}
            tr.stats.sac.evlo=evlo
            tr.stats.sac.evla=evla
            tr.stats.sac.b=tb
            stla, elev, stlo=subdset.coordinates.values()
            # # # stlo=self.waveforms[station_id].coordinates['longitude']
            # # # stla=self.waveforms[station_id].coordinates['latitude']
            tr.stats.sac.stlo=stlo
            tr.stats.sac.stla=stla
            if (abs(evlo-stlo) < 0.01) and (abs(evla-stla) < 0.01):
                warnings.warn('Skipping aftan for: '+ station_id, UserWarning, stacklevel=1)
                continue
            stacode = tr.stats.station
            if prephdir !=None:
                phvelname = prephdir + "/E000.%s.pre" %stacode
            else:
                phvelname =''
            # aftan analysis
            ntrace=ses3dtrace(tr.data, tr.stats)
            ntrace.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin,
                vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax, tresh=inftan.tresh,
                ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, phvelname=phvelname)
            if verbose:
                print 'aftan analysis for', station_id#, ntrace.stats.sac.dist
            station_id_aux=tr.stats.network+tr.stats.station # station_id for auxiliary data("SW4AAA"), not the diference with station_id "SW4.AAA"
            # save aftan results to ASDF dataset
            if basic1:
                parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': ntrace.ftanparam.nfout1_1,
                        'knetwk': tr.stats.network, 'kstnm': tr.stats.station}
                self.add_auxiliary_data(data=ntrace.ftanparam.arr1_1, data_type='DISPbasic1', path=station_id_aux, parameters=parameters)
            if basic2:
                parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'Np': ntrace.ftanparam.nfout2_1,
                        'knetwk': tr.stats.network, 'kstnm': tr.stats.station}
                self.add_auxiliary_data(data=ntrace.ftanparam.arr2_1, data_type='DISPbasic2', path=station_id_aux, parameters=parameters)
            if inftan.pmf:
                if pmf1:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': ntrace.ftanparam.nfout1_2,
                        'knetwk': tr.stats.network, 'kstnm': tr.stats.station}
                    self.add_auxiliary_data(data=ntrace.ftanparam.arr1_2, data_type='DISPpmf1', path=station_id_aux, parameters=parameters)
                if pmf2:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'Np': ntrace.ftanparam.nfout2_2,
                        'knetwk': tr.stats.network, 'kstnm': tr.stats.station}
                    self.add_auxiliary_data(data=ntrace.ftanparam.arr2_2, data_type='DISPpmf2', path=station_id_aux, parameters=parameters)
            if outdir != None:
                foutPR=outdir+"/"+station_id
                tr.ftanparam.writeDISP(foutPR)
        print 'End aftan analysis!'
        return
    
    def aftanMP(self, outdir, deletedisp=True, channel='BXZ', tb=-13.5, inftan=InputFtanParam(), basic1=True, basic2=False,
            pmf1=False, pmf2=False, prephdir=None, subsize=1000):
        """ aftan analysis for ASDF Dataset using multiprocessing module
        Code Notes:
        I tried to use multiprocessing.Manager to define a list shared by all the process and every lock the process when writing to the shared list,
        but unfortunately this somehow doesn't work. As a result, I write this aftan with multiprocessing.Pool, it only speed up about twice compared
        with single processor version aftan.
        """
        print 'Preparing data for aftan analysis (MP)!'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        try:
            evlo=self.events.events[0].origins[0].longitude
            evla=self.events.events[0].origins[0].latitude
        except:
            raise ValueError('No event specified to the datasets!')
        inputStream=[]
        knetwkLst=np.array([])
        kstnmLst=np.array([])
        for station_id in self.waveforms.list():
            # Get data from ASDF dataset
            # # # tr=self.waveforms[station_id].ses3d_raw.select(channel=channel)[0]
            subdset = self.waveforms[station_id]
            tr=subdset.ses3d_raw.select(channel=channel)[0]
            tr.stats.sac={}
            tr.stats.sac.evlo=evlo
            tr.stats.sac.evla=evla
            tr.stats.sac.b=tb
            stla, elev, stlo=subdset.coordinates.values()
            # # # stlo=self.waveforms[station_id].coordinates['longitude']
            # # # stla=self.waveforms[station_id].coordinates['latitude']
            tr.stats.sac.stlo=stlo 
            tr.stats.sac.stla=stla
            if (abs(evlo-stlo) < 0.01) and (abs(evla-stla) < 0.01):
                warnings.warn('Will skip aftan for: '+ station_id+' '+str(stla)+' '+str(stlo), UserWarning, stacklevel=1)
                continue
            ntrace=ses3dtrace(tr.data, tr.stats)
            inputStream.append(ntrace)
            knetwkLst=np.append(knetwkLst, tr.stats.network)
            kstnmLst=np.append(kstnmLst, tr.stats.station)
        # aftan analysis
        # Note: the input stream is divided into several subgroups,
        # since the run can be locked if the number of input trace is large
        #
        print 'Start aftan analysis (MP)!'
        if len(inputStream) > subsize:
            Nsub = int(len(inputStream)/subsize)
            for isub in xrange(Nsub):
                print isub,'in',Nsub
                cstream=inputStream[isub*subsize:(isub+1)*subsize]
                AFTAN = partial(aftan4mp, outdir=outdir, inftan=inftan, prephdir=prephdir)
                pool = multiprocessing.Pool()
                pool.map_async(AFTAN, cstream) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cstream=inputStream[(isub+1)*subsize:]
            AFTAN = partial(aftan4mp, outdir=outdir, inftan=inftan, prephdir=prephdir)
            pool = multiprocessing.Pool()
            pool.map_async(AFTAN, cstream) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            AFTAN = partial(aftan4mp, outdir=outdir, inftan=inftan, prephdir=prephdir)
            pool = multiprocessing.Pool()
            pool.map_async(AFTAN, inputStream) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        print 'End of aftan analysis  ( MP ) !'
        print 'Reading aftan results into ASDF Dataset!'
        for i in xrange(knetwkLst.size):
            # Get data from ASDF dataset
            station_id=knetwkLst[i]+'.'+kstnmLst[i]
            station_id_aux=knetwkLst[i]+kstnmLst[i]
            try:
                f10=np.load(outdir+'/'+station_id+'_1_DISP.0.npz')
                f11=np.load(outdir+'/'+station_id+'_1_DISP.1.npz')
                f20=np.load(outdir+'/'+station_id+'_2_DISP.0.npz')
                f21=np.load(outdir+'/'+station_id+'_2_DISP.1.npz')
            except IOError:
                print 'NO aftan results',station_id
                continue
            print 'Reading aftan results',station_id
            if deletedisp:
                os.remove(outdir+'/'+station_id+'_1_DISP.0.npz')
                os.remove(outdir+'/'+station_id+'_1_DISP.1.npz')
                os.remove(outdir+'/'+station_id+'_2_DISP.0.npz')
                os.remove(outdir+'/'+station_id+'_2_DISP.1.npz')
            arr1_1=f10['arr_0']
            nfout1_1=f10['arr_1']
            arr2_1=f11['arr_0']
            nfout2_1=f11['arr_1']
            arr1_2=f20['arr_0']
            nfout1_2=f20['arr_1']
            arr2_2=f21['arr_0']
            nfout2_2=f21['arr_1']
            if basic1:
                parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': nfout1_1,
                        'knetwk': str(knetwkLst[i]), 'kstnm': str(kstnmLst[i])}
                self.add_auxiliary_data(data=arr1_1, data_type='DISPbasic1', path=station_id_aux, parameters=parameters)
            if basic2:
                parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'Np': nfout2_1,
                        'knetwk': str(knetwkLst[i]), 'kstnm': str(kstnmLst[i])}
                self.add_auxiliary_data(data=arr2_1, data_type='DISPbasic2', path=station_id_aux, parameters=parameters)
            if inftan.pmf:
                if pmf1:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': nfout1_2,
                        'knetwk': str(knetwkLst[i]), 'kstnm': str(kstnmLst[i])}
                    self.add_auxiliary_data(data=arr1_2, data_type='DISPpmf1', path=station_id_aux, parameters=parameters)
                if pmf2:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'Np': nfout2_2,
                        'knetwk': str(knetwkLst[i]), 'kstnm': str(kstnmLst[i])}
                    self.add_auxiliary_data(data=arr2_2, data_type='DISPpmf2', path=station_id_aux, parameters=parameters)
        return
    
    def select_data(self, outfname, stafile, sacflag=True, channel=['BXZ'], data_type='DISPbasic1' ):
        """ Select data from ASDF Dataset
        =================================================================================
        Input Parameters:
        outfname    - output ASDF file name
        stafile     - station list file name
        sacflag     - select sac data or not
        compindex   - component index in waveforms path (default = np.array([0]))
        data_type   - dispersion data type (default = DISPbasic1, basic aftan results)
        Output:
        Ndbase      - a new ASDF data file
        =================================================================================
        """
        SLst = stations.StaLst()
        SLst.read(stafile=stafile)
        StaInv=SLst.GetInventory()
        Ndbase=ses3dASDF(outfname)
        Ndbase.add_stationxml(StaInv)
        Ndbase.add_quakeml(self.events)
        disptypelst=['DISPbasic1', 'DISPbasic2', 'DISPpmf1', 'DISPpmf2']
        for sta in SLst.stations:
            station_id=sta.network+'.'+sta.stacode
            station_id_aux=sta.network+sta.stacode
            if sacflag==True:
                try:
                    for chan in channel:
                        tr=self.waveforms[station_id].sw4_raw.select(channel=chan)[0]
                        Ndbase.add_waveforms(tr, tag='sw4_raw')
                except:
                    print 'No sac data for:',station_id,'!'
            if data_type!='All' and data_type !='all':
                try:
                    data=self.auxiliary_data[data_type][station_id_aux].data.value
                    parameters=self.auxiliary_data[data_type][station_id_aux].parameters
                    Ndbase.add_auxiliary_data(data=data, data_type=data_type, path=station_id_aux, parameters=parameters)
                except:
                    print 'No', data_type, 'data for:', station_id, '!'
            else:
                for dispindex in disptypelst:
                    try:
                        data=self.auxiliary_data[dispindex][station_id_aux].data.value
                        parameters=self.auxiliary_data[dispindex][station_id_aux].parameters
                        Ndbase.add_auxiliary_data(data=data, data_type=dispindex, path=station_id_aux, parameters=parameters)
                    except:
                        print 'No', dispindex, 'data for:', station_id, '!'
        return Ndbase
    
    def interpDisp(self, data_type='DISPbasic1', pers=np.array([10., 15., 20., 25.]), verbose=True):
        """ Interpolate dispersion curve for a given period array.
        ================================================================================
        Input Parameters:
        data_type   - dispersion data type (default = DISPbasic1, basic aftan results)
        pers        - period array
        
        Output:
        self.auxiliary_data.DISPbasic1interp, self.auxiliary_data.DISPbasic2interp,
        self.auxiliary_data.DISPpmf1interp, self.auxiliary_data.DISPpmf2interp
        ================================================================================
        """
        staidLst=self.auxiliary_data[data_type].list()
        for staid in staidLst:
            subdset = self.auxiliary_data[data_type][staid]
            data=subdset.data.value
            index=subdset.parameters
            knetwk=str(subdset.parameters['knetwk'])
            kstnm=str(subdset.parameters['kstnm'])
            if verbose:
                print 'Interpolating dispersion curve for '+ knetwk + kstnm
            outindex={ 'To': 0, 'Vgr': 1, 'Vph': 2,  'amp': 3, 'inbound': 4, 'Np': pers.size, 'knetwk': knetwk, 'kstnm': kstnm }
            Np=index['Np']
            if Np < 5:
                warnings.warn('Not enough datapoints for: '+ knetwk+'.'+kstnm, UserWarning, stacklevel=1)
                continue
                # print 'Not enough datapoints for: '+ knetwk+'.'+kstnm
            obsT=data[index['To']][:Np]
            Vgr=np.interp(pers, obsT, data[index['Vgr']][:Np] )
            Vph=np.interp(pers, obsT, data[index['Vph']][:Np] )
            amp=np.interp(pers, obsT, data[index['amp']][:Np] )
            inbound=(pers > obsT[0])*(pers < obsT[-1])*1
            interpdata=np.append(pers, Vgr)
            interpdata=np.append(interpdata, Vph)
            interpdata=np.append(interpdata, amp)
            interpdata=np.append(interpdata, inbound)
            interpdata=interpdata.reshape(5, pers.size)
            self.add_auxiliary_data(data=interpdata, data_type=data_type+'interp', path=staid, parameters=outindex)
        return
    
    def get_field(self, data_type='DISPbasic1', fieldtype='Vgr', pers=np.array([10.]), outdir=None, distflag=True, verbose=True ):
        """ Get the field data
        =======================================================================================
        Input Parameters:
        data_type   - dispersion data type (default = DISPbasic1, basic aftan results)
        fieldtype   - field data type( Vgr, Vph, Amp)
        pers        - period array
        outdir      - directory for txt output
        distflag    - whether to output distance or not
        Output:
        self.auxiliary_data.FieldDISPbasic1interp, self.auxiliary_data.FieldDISPbasic2interp,
        self.auxiliary_data.FieldDISPpmf1interp, self.auxiliary_data.FieldDISPpmf2interp
        =======================================================================================
        """
        data_type=data_type+'interp'
        tempdict={'Vgr': 'Tgr', 'Vph': 'Tph', 'amp': 'Amp', 'ms':'Ms'}
        if distflag:
            outindex={ 'longitude': 0, 'latitude': 1, tempdict[fieldtype]: 2,  'dist': 3 }
        else:
            outindex={ 'longitude': 0, 'latitude': 1, tempdict[fieldtype]: 2 }
        staidLst=self.auxiliary_data[data_type].list()
        evlo=self.events.events[0].origins[0].longitude
        evla=self.events.events[0].origins[0].latitude
        for per in pers:
            FieldArr=np.array([])
            Nfp=0
            for staid in staidLst:
                subdset = self.auxiliary_data[data_type][staid]
                data=subdset.data.value
                index=subdset.parameters
                knetwk=str(subdset.parameters['knetwk'])
                kstnm=str(subdset.parameters['kstnm'])
                station_id=knetwk+'.'+kstnm
                obsT=data[index['To']]
                if verbose:
                    print 'Getting field data from '+ station_id
                if fieldtype=='ms':
                    outdata=data[index['Vgr']]
                else:
                    outdata=data[index[fieldtype]]
                inbound=data[index['inbound']]
                fieldpoint=outdata[obsT==per]
                if fieldpoint == np.nan or fieldpoint<=0:
                    print station_id+' has invalid value'+' T='+str(per)+'s'
                    continue
                # print fieldpoint
                inflag=inbound[obsT==per]
                if fieldpoint.size==0:
                    print 'No datapoint for'+ station_id+' T='+per+'s in interpolated disp dataset!'
                    continue
                if inflag == 0:
                    print 'Datapoint out of bound: '+ knetwk+'.'+kstnm+' T='+str(per)+'s!'
                    continue
                if fieldtype=='ms':
                    subdset=self.waveforms[station_id]
                    tr=subdset.ses3d_raw[0]
                    tr.stats.sac={}
                    tr.stats.sac.evlo=evlo
                    tr.stats.sac.evla=evla
                    stla, elev, stlo=subdset.coordinates.values()
                    dist, az, baz=obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                    distance = dist/1000.
                    tr.stats.sac.dist=distance
                    ntrace=ses3dtrace(tr.data, tr.stats)
                    try:
                        ab, Ms=ntrace.get_ms(Vgr=fieldpoint)
                    except:
                        continue
                    fieldpoint=Ms
                else:
                    stla, elev, stlo=self.waveforms[station_id].coordinates.values()
                    dist, az, baz=obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                    distance = dist/1000.
                if distance == 0.:
                    continue
                FieldArr=np.append(FieldArr, stlo)
                FieldArr=np.append(FieldArr, stla)
                if fieldtype=='Vgr' or fieldtype=='Vph':
                    fieldpoint=distance/fieldpoint
                FieldArr=np.append(FieldArr, fieldpoint)
                if distflag:
                    FieldArr=np.append(FieldArr, distance)
                Nfp+=1
            if distflag:
                FieldArr=FieldArr.reshape( Nfp, 4)
            else:
                FieldArr=FieldArr.reshape( Nfp, 3)
            if outdir!=None:
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                txtfname=outdir+'/'+tempdict[fieldtype]+'_'+str(per)+'.txt'
                header = 'evlo='+str(evlo)+' evla='+str(evla)
                np.savetxt( txtfname, FieldArr, fmt='%g', header=header )
            self.add_auxiliary_data(data=FieldArr, data_type='Field'+data_type, path=tempdict[fieldtype]+str(int(per)), parameters=outindex)
        return
    
    def get_all_field(self, data_type='DISPbasic1', pers=np.array([10.]), outdir=None, distflag=True, verbose=True ):
        """ Get the field data
        =======================================================================================
        Input Parameters:
        data_type   - dispersion data type (default = DISPbasic1, basic aftan results)
        pers        - period array
        outdir      - directory for txt output
        distflag    - whether to output distance or not
        Output:
        self.auxiliary_data.FieldDISPbasic1interp, self.auxiliary_data.FieldDISPbasic2interp,
        self.auxiliary_data.FieldDISPpmf1interp, self.auxiliary_data.FieldDISPpmf2interp
        =======================================================================================
        """
        data_type=data_type+'interp'
        if distflag:
            outindexPh={ 'longitude': 0, 'latitude': 1, 'Tph': 2,  'dist': 3 }
            outindexGr={ 'longitude': 0, 'latitude': 1, 'Tgr': 2,  'dist': 3 }
            outindexAmp={ 'longitude': 0, 'latitude': 1,'Amp': 2,  'dist': 3 }
        else:
            outindexPh={ 'longitude': 0, 'latitude': 1, 'Tph': 2}
            outindexGr={ 'longitude': 0, 'latitude': 1, 'Tgr': 2}
            outindexAmp={ 'longitude': 0, 'latitude': 1,'Amp': 2}
        staidLst=self.auxiliary_data[data_type].list()
        evlo=self.events.events[0].origins[0].longitude
        evla=self.events.events[0].origins[0].latitude
        for per in pers:
            FieldPh=np.array([])
            FieldGr=np.array([])
            FieldAmp=np.array([])
            Nfp=0
            for staid in staidLst:
                subdset = self.auxiliary_data[data_type][staid]
                data=subdset.data.value
                index=subdset.parameters
                knetwk=str(subdset.parameters['knetwk'])
                kstnm=str(subdset.parameters['kstnm'])
                if verbose:
                    print 'Getting all field data from '+ staid
                station_id=knetwk+'.'+kstnm
                obsT=data[index['To']]
                dataPh=data[index['Vph']]
                dataGr=data[index['Vgr']]
                dataAmp=data[index['amp']]
                inbound=data[index['inbound']]
                perindex = np.where(obsT==per)[0]
                if perindex.size == 0:
                    print 'No datapoint for'+ station_id+' T='+per+'s in interpolated disp dataset!'
                    continue
                inflag=inbound[perindex]
                if inflag == 0:
                    print 'Datapoint out of bound: '+ knetwk+'.'+kstnm+' T='+str(per)+'s!'
                    continue
                Vph=dataPh[perindex]
                Vgr=dataGr[perindex]
                Amp=dataAmp[perindex]
                if Vph == np.nan or Vph<=0 or Vgr == np.nan or Vgr<=0 or Amp == np.nan or Amp<=0:
                    warnings.warn(station_id+' has invalid value'+' T='+str(per)+'s', UserWarning, stacklevel=1)
                    continue
                stla, elev, stlo=self.waveforms[station_id].coordinates.values()
                # stlo=self.waveforms[station_id].coordinates['longitude']
                # stla=self.waveforms[station_id].coordinates['latitude']
                dist, az, baz=obspy.geodetics.base.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                distance = dist/1000.
                if distance == 0.:
                    continue
                FieldPh=np.append(FieldPh, stlo)
                FieldPh=np.append(FieldPh, stla)
                FieldPh=np.append(FieldPh, distance/Vph)
                FieldGr=np.append(FieldGr, stlo)
                FieldGr=np.append(FieldGr, stla)
                FieldGr=np.append(FieldGr, distance/Vgr)
                FieldAmp=np.append(FieldAmp, stlo)
                FieldAmp=np.append(FieldAmp, stla)
                FieldAmp=np.append(FieldAmp, Amp)
                if distflag:
                    FieldPh=np.append(FieldPh, distance)
                    FieldGr=np.append(FieldGr, distance)
                    FieldAmp=np.append(FieldAmp, distance) 
                Nfp+=1
            if distflag:
                FieldPh=FieldPh.reshape( Nfp, 4)
                FieldGr=FieldGr.reshape( Nfp, 4)
                FieldAmp=FieldAmp.reshape( Nfp, 4)
            else:
                FieldPh=FieldPh.reshape( Nfp, 3)
                FieldGr=FieldGr.reshape( Nfp, 3)
                FieldAmp=FieldAmp.reshape( Nfp, 3)
                
            if outdir!=None:
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                txtfnamePh=outdir+'/Tph_'+str(per)+'.txt'
                txtfnameGr=outdir+'/Tgr_'+str(per)+'.txt'
                txtfnameAmp=outdir+'/Amp_'+str(per)+'.txt'
                header = 'evlo='+str(evlo)+' evla='+str(evla)
                np.savetxt( txtfnamePh, FieldPh, fmt='%g', header=header )
                np.savetxt( txtfnameGr, FieldGr, fmt='%g', header=header )
                np.savetxt( txtfnameAmp, FieldAmp, fmt='%g', header=header )
            self.add_auxiliary_data(data=FieldPh, data_type='Field'+data_type, path='Tph'+str(int(per)), parameters=outindexPh)
            self.add_auxiliary_data(data=FieldGr, data_type='Field'+data_type, path='Tgr'+str(int(per)), parameters=outindexGr)
            self.add_auxiliary_data(data=FieldAmp, data_type='Field'+data_type, path='Amp'+str(int(per)), parameters=outindexAmp)
        return
    
    def write2sac(self, staid=None, lon=None, lat=None, outdir='.', SLst=None, channel='BXZ'):
        if staid==None and (lon==None or lat==None):
            raise ValueError('Error Input')
        evlo=self.events.events[0].origins[0].longitude
        evla=self.events.events[0].origins[0].latitude
        if staid!=None:
            tr=self.waveforms[staid].ses3d_raw.select(channel=channel)[0]
            lat, elev, lon=self.waveforms[staid].coordinates.values()
        elif isinstance(SLst, stations.StaLst):
            for sta in SLst.stations:
                if sta.lon!=lon or sta.lat!=lat:
                    continue
                else:
                    staid=sta.network+'.'+sta.stacode
                    tr=self.waveforms[staid].ses3d_raw.select(channel=channel)[0]
                    break
        else:
            for staid in self.waveforms.list():
                # Get data from ASDF dataset
                subdset = self.waveforms[staid]
                stla, elev, stlo=subdset.coordinates.values()
                if stlo!=lon or stla!=lat:
                    continue
                else:
                    tr=subdset.ses3d_raw.select(channel=channel)[0]
                    break
        sacTr=obspy.io.sac.sactrace.SACTrace.from_obspy_trace(tr)
        sacTr.evlo=evlo
        sacTr.evla=evla
        sacTr.stlo=lon
        sacTr.stla=lat
        outsacfname=outdir+'/'+staid+'.SAC'
        sacTr.write(outsacfname)
        return
    
    def get_trace(self, staid=None, lon=None, lat=None, outdir='.', SLst=None, channel='BXZ'):
        if staid==None and (lon==None or lat==None):
            raise ValueError('Error Input')
        evlo=self.events.events[0].origins[0].longitude
        evla=self.events.events[0].origins[0].latitude
        if staid!=None:
            tr=self.waveforms[staid].ses3d_raw.select(channel=channel)[0]
            lat, elev, lon=self.waveforms[staid].coordinates.values()
        elif isinstance(SLst, stations.StaLst):
            for sta in SLst.stations:
                if sta.lon!=lon or sta.lat!=lat:
                    continue
                else:
                    staid=sta.network+'.'+sta.stacode
                    tr=self.waveforms[staid].ses3d_raw.select(channel=channel)[0]
                    break
        else:
            for staid in self.waveforms.list():
                # Get data from ASDF dataset
                subdset = self.waveforms[staid]
                stla, elev, stlo=subdset.coordinates.values()
                if stlo!=lon or stla!=lat:
                    continue
                else:
                    tr=subdset.ses3d_raw.select(channel=channel)[0]
                    break
        tr.stats.coordinates = AttribDict({
            'latitude': lat,
            'elevation': lon,
            'longitude': 0.0})
        return tr
    
    
    
    def get_dist_az_baz(self, staid=None, stlo=None, stla=None):
        if staid==None and (lon==None or lat==None):
            raise ValueError('Error Input')
        evlo=self.events.events[0].origins[0].longitude
        evla=self.events.events[0].origins[0].latitude
        if staid!=None:
            stla, elev, stlo=self.waveforms[staid].coordinates.values()
        dist, az, baz=obspy.geodetics.base.gps2dist_azimuth( stla, stlo, evla, evlo) # distance is in m
        return dist/1000., az, baz
    
    def PlotStreamsDistance(self, compindex=0, norm_method='trace'):
        try:
            evlo=self.events.events[0].origins[0].longitude
            evla=self.events.events[0].origins[0].latitude
        except:
            raise ValueError('No event specified to the datasets!')
        Str4Plot=obspy.core.Stream()

        for station_id in self.waveforms.list():
            # Get data from ASDF dataset
            tr=self.waveforms[station_id].sw4_raw[compindex]
            stlo=self.waveforms[station_id].coordinates['longitude']*100.
            stla=self.waveforms[station_id].coordinates['latitude']*100.
            station_id_aux=tr.stats.network+tr.stats.station
            if stlo > 1500.:
                tr.stats.distance = np.sqrt( (stlo-evlo)**2 + (stla-evla)**2 ) *1000.
            else:
                tr.stats.distance =- np.sqrt( (stlo-evlo)**2 + (stla-evla)**2 ) *1000.
            if stlo%100 !=0 or abs(stlo-1500.) < 200:
                # print tr.stats.distance/1000.
                continue
            print station_id, tr.stats.distance
            Str4Plot.append(tr)

        Str4Plot.plot(type='section', norm_method='stream', recordlength=600, alpha=1.0, scale=4.0, offset_min=-1500000, offset_max=1500000,
                    linewidth = 2.5 )
        # plt.show()
        return
        
    def get_ms(self, SLst, outdir, inftan=InputFtanParam(), mindist=-float('inf'), maxdist=float('inf'), minaz=-1, maxaz=361.,
               inaz=None, daz=.5, channel='BXZ', plottype='azimuth', prephdir=None, period=10., wfactor=1.):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        try:
            evlo=self.events.events[0].origins[0].longitude
            evla=self.events.events[0].origins[0].latitude
        except:
            raise ValueError('No event specified to the datasets!')
        if inaz!=None:
            minaz=inaz-daz
            maxaz=inaz+daz
        else:
            minaz=minaz
            maxaz=maxaz 
        newSLst=stations.StaLst()
        azArr=np.array([])
        distArr=np.array([])
        abArr=np.array([])
        MsArr=np.array([])
        vgrArr=np.array([])
        ampArr=np.array([])
        for sta in SLst.stations:
            dist, az, baz=obspy.geodetics.gps2dist_azimuth( evla, evlo, sta.lat, sta.lon) # distance is in m
            dist=dist/1000.
            if dist<mindist or dist > maxdist or az<minaz or az>maxaz:
                continue
            station_id=sta.network+'.'+sta.stacode
            print 'Getting Ms for',station_id
            subdset = self.waveforms[station_id]
            tr=subdset.ses3d_raw.select(channel=channel)[0]
            tr.stats.sac={}
            tr.stats.sac.evlo=evlo
            tr.stats.sac.evla=evla
            tr.stats.sac.b=0.
            stla, elev, stlo=subdset.coordinates.values()
            tr.stats.sac.stlo=stlo
            tr.stats.sac.stla=stla
            if (abs(evlo-stlo) < 0.01) and (abs(evla-stla) < 0.01):
                warnings.warn('Skipping aftan for: '+ station_id, UserWarning, stacklevel=1)
                continue
            stacode = tr.stats.station
            if prephdir !=None:
                phvelname = prephdir + "/E000.%s.pre" %stacode
            else:
                phvelname =''
            # aftan analysis
            ntrace=ses3dtrace(tr.data, tr.stats)
            ntrace.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin,
                vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax, tresh=inftan.tresh,
                ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, phvelname=phvelname)
            if ntrace.ftanparam.nfout2_2==0:
                vmin=2.2
                print 'changing vmin:',station_id
                while(ntrace.ftanparam.nfout2_2==0):
                    ntrace.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=vmin,
                        vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax, tresh=inftan.tresh,
                        ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, phvelname=phvelname)
                    vmin+=0.1
                    if vmin>3.5:
                        break
            try:
                ab, ms, vgr, amp=ntrace.get_ms(period=period, wfactor=wfactor)
            except:
                print 'skipping',station_id
                continue
            newSLst.append(sta)
            azArr=np.append(azArr, az)
            distArr=np.append(distArr, dist)
            abArr=np.append(abArr, ab)
            ampArr=np.append(ampArr, amp)
            MsArr=np.append(MsArr, ms)
            vgrArr=np.append(vgrArr, vgr)
        if plottype=='dist':
            plt.subplot(211)
            plt.plot(azArr, MsArr, '^')
            plt.subplot(212)
            plt.plot(azArr, abArr, '^')
            plt.show()
            outfname=outdir+'/dist_'+str(mindist)+'_'+str(maxdist)+'_amp_Ms.lst'
            outArr=np.append(azArr, abArr)
            outArr=np.append(outArr, MsArr)
            outArr=np.append(outArr, ampArr)
            outArr=np.append(outArr, vgrArr)
            outArr=np.append(outArr, distArr)
            outArr=outArr.reshape(6, MsArr.size)
            outArr=outArr.T
            header='az ab Ms amp Vgr dist'
            np.savetxt(outfname, outArr, fmt='%g', header=header)
            newSLst.write(outdir+'/dist_'+str(mindist)+'_'+str(maxdist))
        elif plottype=='azimuth':
            plt.subplot(211)
            plt.plot(distArr, MsArr, '^')
            plt.subplot(212)
            plt.plot(distArr, abArr, '^')
            plt.show()
            outfname=outdir+'/az_'+str(minaz)+'_'+str(maxaz)+'_amp_Ms.lst'
            outArr=np.append(distArr, abArr)
            outArr=np.append(outArr, MsArr)
            outArr=np.append(outArr, ampArr)
            outArr=np.append(outArr, vgrArr)
            outArr=np.append(outArr, azArr)
            outArr=outArr.reshape(6, MsArr.size)
            outArr=outArr.T
            header='dist ab Ms amp Vgr az'
            np.savetxt(outfname, outArr, fmt='%g', header=header)
            newSLst.write(outdir+'/az_'+str(minaz)+'_'+str(maxaz))
        
            
            
            
        

def aftan4mp(nTr, outdir, inftan, prephdir):
    # print 'aftan analysis for', nTr.stats.network, nTr.stats.station#, i.value#, ntrace.stats.sac.dist
    if prephdir !=None:
        phvelname = prephdir + "/E000.%s.pre" %nTr.stats.station
    else:
        phvelname =''
    nTr.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin,
                vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax, tresh=inftan.tresh,
                ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, phvelname=phvelname)
    foutPR=outdir+'/'+nTr.stats.network+'.'+nTr.stats.station
    nTr.ftanparam.writeDISPbinary(foutPR)
    return
            
