
import numpy as np
import obspy
import scipy.signal
import matplotlib.pyplot as plt
import os
import subprocess
from lasif import rotations

class STF(obspy.core.trace.Trace):
    """
    An object inherited from obspy Trace to handle source time function
    """
    def BruneSignal(self, dt, npts, tauR=1.):
        """
        Reference:
        Brune, J.N., 1970. Tectonic stress and the spectra of seismic shear waves from earthquakes.
            Journal of geophysical research, 75(26), pp.4997-5009.
        """
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        self.data=(1.-(1.+time/tauR)*np.exp(-time/tauR))*0.5 * (np.sign(time) + 1.)
        return
    
    def HaskellSignal(self, dt, npts, k=31.6, B=0.24):
        """
        References:
        Haskell, N.A., 1967. Analytic approximation for the elastic radiation from a contained underground explosion.
            Journal of Geophysical Research, 72(10), pp.2583-2587.
        """
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        KT=k*time
        self.data=1.-np.exp( -KT )*( 1.+KT+np.power(KT,2)/2.+np.power(KT,3)/6-B*np.power(KT,4))
        return
    
    def vonSeggernSignal(self, dt, npts, k=16.8, B=2.04):
        """
        References:
        von Seggern, D. and Blandford, R., 1972. Source time functions and spectra for underground nuclear explosions.
            Geophysical Journal International, 31(1-3), pp.83-97.
        """
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        KT=k*time
        self.data=1.-np.exp( -KT )*( 1.+KT-B*np.power(KT,2) )
        return
    
    def HelmbergerSignal(self, dt, npts, k=16.8, B=2.04):
        """
        References:
        Helmberger, D.V. and Hadley, D.M., 1981. Seismic source functions and attenuation from local and teleseismic observations
            of the NTS events JORUM and HANDLEY. Bulletin of the Seismological Society of America, 71(1), pp.51-67.
        """
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        KT=k*time
        self.data=1.-np.exp( -KT )*( 1.+KT+np.power(KT,2)/2.-B*np.power(KT,3) )
        return
    
    def StepSignal(self, dt, npts):
        """Step function
        """
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        self.data=np.ones(time.size)
        return
    
    def RickerWavelet(self, dt, npts, fpeak):
        """Ricker wavelet
        """
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        a=1./(fpeak*math.pi)
        self.data=scipy.signal.ricker(npts, a=a)
        return
    
    def GaussianSignal(self, dt, npts, fc, t0=None):
        """Gaussian signal defined in sw4 manual(p 17)
        """
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        if t0==None:
            Nshift=np.int(6./fc/dt) - 1
            t0=Nshift *dt
        else:
            Nshift=np.int(t0/dt) - 1
            t0=Nshift *dt
        tempSig=fc/np.sqrt(2.*np.pi) *np.exp(- ( fc*(time-t0) )**2 / 2. )
        self.data=tempSig
        self.fcenter=fc
        return

    def RickerIntSignal(self, dt, npts, fc, t0=None):
        """RickerInt signal defined in sw4 manual(p 18)
        """
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        if t0==None:
            Nshift=np.int(1.35/fc/dt) - 1
            t0=Nshift *dt
        else:
            Nshift=np.int(t0/dt) - 1
            t0=Nshift *dt
        tempSig=(time - t0) *np.exp(- (np.pi*fc*(time-t0) )**2 )
        self.data=tempSig
        self.fcenter=fc
        return 
    
    def dofft(self, diff=True):
        """Do FFT to get spectrum
        """
        self.diff=diff
        npts=self.data.size
        Ns=1<<(npts-1).bit_length()
        INput = np.zeros((Ns), dtype=complex)
        OUTput = np.zeros((Ns), dtype=complex)
        if diff==True:
            tempTr=self.copy()
            tempTr.differentiate()
            INput[:npts]=tempTr.data
        else:
            INput[:npts]=self.data
        nhalf=Ns/2+1
        OUTput = np.fft.fft(INput)
        OUTput[nhalf:]=0
        OUTput[0]/=2
        OUTput[nhalf-1]=OUTput[nhalf-1].real+0.j
        self.hf = OUTput
        self.freq = np.fft.fftfreq(len(self.hf), self.stats.delta)
        return
    
    def plotfreq(self):
        """Plot spectrum
        """
        try:
            freq=self.freq
            hf=self.hf
        except:
            self.dofft()
            freq=self.freq
            hf=self.hf
        plt.semilogx(freq, np.abs(hf), lw=3)
        plt.xlabel('frequency [Hz]')
        if self.diff == True:
            plt.title('Time derivative source time function (frequency domain)')
        else:
            plt.title('source time function (frequency domain)')
        plt.show()
        return
    
    def plotstf(self, fmax=None):
        """Plot source time function and its corresponding time derivative
        """
        if fmax==None:
            try:
                fmax=self.fcenter*2.5
            except:
                raise AttributeError('Maximum frequency not specified!')
        ax=plt.subplot(411)
        ax.plot(np.arange(self.stats.npts)*self.stats.delta, self.data, 'k-', lw=3)
        plt.xlim(0,100./fmax)
        plt.xlabel('time [s]')
        plt.title('source time function (time domain)')
        ax=plt.subplot(412)
        self.dofft()
        self.plotfreq()
        # plt.xlim(fmin/5.,fmax*5.)
        plt.xlabel('frequency [Hz]')
        plt.title('source time function (frequency domain)')
        
        ax=plt.subplot(413)
        outSTF=self.copy()
        self.differentiate()
        ax.plot(np.arange(self.stats.npts)*self.stats.delta, self.data, 'k-', lw=3)
        plt.xlim(0,100./fmax)
        plt.xlabel('time [s]')
        plt.title('Derivative of source time function (time domain)')
        
        ax=plt.subplot(414)
        self.dofft()
        self.plotfreq()
        # plt.xlim(fmin/5.,fmax*5.)
        plt.xlabel('frequency [Hz]')
        plt.title('Derivative of source time function (frequency domain)')
        
        plt.show()
    
    
class ses3dCatalog(obspy.core.event.Catalog):
    """Catalog object inherited from obspy catalog for ses3d preprocessing
    """
    def add_event(self, longitude, latitude, depth, event_type='earthquake', focalmechanism=None):
        """Add event to the catalog
        """
        try:
            tensor = focalmechanism.moment_tensor.tensor
            inputstr="mopad describe %e,%e,%e,%e,%e,%e" %(tensor.m_rr, tensor.m_tt, tensor.m_pp, tensor.m_rt, tensor.m_rp, tensor.m_tp)
            shMopad='./tempmopad.sh'
            with open(shMopad, 'wb') as f:
                f.writelines(inputstr)
            out=subprocess.check_output(['bash', shMopad])
            m0 = float(out.split()[4])
            os.remove(shMopad)
            Mw = 2.0 / 3.0 * (np.log10(m0) - 9.1)
            Magnitude=obspy.core.event.magnitude.Magnitude(magnitude_type='Mw', mag=Mw)
            origin=obspy.core.event.origin.Origin(longitude=longitude, latitude=latitude, depth=depth, time=obspy.core.utcdatetime.UTCDateTime() )
            event=obspy.core.event.event.Event(origins=[origin], event_type=event_type, focal_mechanisms=[focalmechanism], magnitudes=[Magnitude] )
        except:
            origin=obspy.core.event.origin.Origin(longitude=longitude, latitude=latitude, depth=depth, time=obspy.core.utcdatetime.UTCDateTime() )
            event=obspy.core.event.event.Event(origins=[origin], event_type=event_type, focal_mechanisms=[focalmechanism])
        self.append(event)
        return 
    
    def add_explosion(self, longitude, latitude, depth, m0):
        """Add explosive event with moment = m0 to catalog
        """
        tensor = obspy.core.event.source.Tensor(m_rr=m0, m_tt=m0, m_pp=m0,
                                                 m_tp=0., m_rt=0., m_rp=0.)
        moment_tensor = obspy.core.event.source.MomentTensor(tensor=tensor, scalar_moment=m0)
        FocalMech = obspy.core.event.source.FocalMechanism(moment_tensor=moment_tensor)
        self.add_event(longitude=longitude, latitude=latitude, depth=depth, event_type='explosion', focalmechanism=FocalMech)
        return
    
    def add_earthquake(self, longitude, latitude, depth, m_rr, m_tt, m_pp, m_tp, m_rt, m_rp):
        """Add event with given moment tensor to catalog
        """
        tensor = obspy.core.event.source.Tensor(m_rr=m_rr, m_tt=m_tt, m_pp=m_pp,
                                                 m_tp=m_tp, m_rt=m_rt, m_rp=m_rp)
        moment_tensor = obspy.core.event.source.MomentTensor(tensor=tensor)
        FocalMech = obspy.core.event.source.FocalMechanism(moment_tensor=moment_tensor)
        self.add_event(longitude=longitude, latitude=latitude, depth=depth, focalmechanism=FocalMech)
        return
    
    def write(self, outdir, config=None, nt=None, dt = None, output_folder=None, ssamp=None, outdisp= None):
        """Write event file to ouput directory 
        """
        try:
            nt = config.number_of_time_steps
            dt = config.time_increment_in_s
            output_folder=config.output_folder
            ssamp=config.displacement_snapshot_sampling
            outdisp = config.output_displacement
        except:
            if nt == None or dt ==None or output_folder == None or ssamp == None or outdisp == None:
                raise ValueError('Wrong input for events writer!')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        eventlst=outdir+'/event_list'
        L=len(self.events)
        lst_template=("{L:<44d}! n_events = number of events\n")
        lst_file = lst_template.format( L = int(L) )
        with open(eventlst, 'wb') as f:
            f.writelines(lst_file)
            for i in xrange(L):
                f.writelines('%d \n' %(i+1))
        event_template = (
            "SIMULATION PARAMETERS ==============================================="
            "===================================\n"
            "{nt:<44d}! nt, number of time steps\n"
            "{dt:<44.6f}! dt in sec, time increment\n"
            "SOURCE =============================================================="
            "===================================\n"
            "{xxs:<44.6f}! xxs, theta-coord. center of source in degrees\n"
            "{yys:<44.6f}! yys, phi-coord. center of source in degrees\n"
            "{zzs:<44.6f}! zzs, source depth in (m)\n"
            "{srctype:<44d}! srctype, 1:f_x, 2:f_y, 3:f_z, 10:M_ij\n"
            "{m_tt:<44.6e}! M_theta_theta\n"
            "{m_pp:<44.6e}! M_phi_phi\n"
            "{m_rr:<44.6e}! M_r_r\n"
            "{m_tp:<44.6e}! M_theta_phi\n"
            "{m_tr:<44.6e}! M_theta_r\n"
            "{m_pr:<44.6e}! M_phi_r\n"
            "OUTPUT DIRECTORY ===================================================="
            "==================================\n"
            "{output_folder}\n"
            "OUTPUT FLAGS ========================================================"
            "==================================\n"
            "{ssamp:<44d}! ssamp, snapshot sampling\n"
            "{output_displacement:<44d}! output_displacement, output displacement "
            "field (1=yes,0=no)")
        
        for i in xrange(L):
            eventname=outdir+'/event_%d' %(i+1)
            with open(eventname, 'wb') as f:
                lat=self.events[i].origins[0].latitude
                lon=self.events[i].origins[0].longitude
                evdp=self.events[i].origins[0].depth*1000.
                m_tt = self.events[i].focal_mechanisms[0].moment_tensor.tensor.m_tt
                m_rr = self.events[i].focal_mechanisms[0].moment_tensor.tensor.m_rr
                m_pp = self.events[i].focal_mechanisms[0].moment_tensor.tensor.m_pp
                m_tp = self.events[i].focal_mechanisms[0].moment_tensor.tensor.m_tp
                m_tr = self.events[i].focal_mechanisms[0].moment_tensor.tensor.m_rt
                m_pr = self.events[i].focal_mechanisms[0].moment_tensor.tensor.m_rp
                
                event_file = event_template.format(
                    nt=int(nt),
                    dt=float(dt),
                    # Colatitude!
                    xxs=rotations.lat2colat(lat),
                    yys=lon,
                    zzs=evdp,
                    srctype=10,
                    m_tt=float(m_tt),
                    m_pp=float(m_pp),
                    m_rr=float(m_rr),
                    m_tp=float(m_tp),
                    m_tr=float(m_tr),
                    m_pr=float(m_pr),
                    output_folder=output_folder,
                    ssamp=int(ssamp),
                    output_displacement=outdisp)
                f.writelines(event_file)
        return
    
    def plotevent(self):
        """Plot event
        """
        if len(self.events) == 1:
            self.events[0].plot(kind=['ortho', 'beachball', 'local'])
        else:
            self.plot(projection='ortho')
        return

                
                
        
        