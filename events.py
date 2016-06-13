
import numpy as np
import obspy
import scipy.signal
import matplotlib.pyplot as plt

class STF(obspy.core.trace.Trace):
    """
    An object inherited from obspy Trace to handle source time function
    """
    
    def BruneSignal(self, dt, npts, tauR=1.):
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        self.data=(1.-(1.+time/tauR)*np.exp(-time/tauR))*0.5 * (np.sign(time) + 1.)
        return
    
    def HaskellSignal(self, dt, npts, k=31.6, B=0.24):
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        KT=k*time
        self.data=1.-np.exp( -KT )*( 1.+KT+np.power(KT,2)/2.+np.power(KT,3)/6-B*np.power(KT,4))
        return
    
    def vonSeggernSignal(self, dt, npts, k=16.8, B=2.04):
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        KT=k*time
        self.data=1.-np.exp( -KT )*( 1.+KT-B*np.power(KT,2) )
        return
    
    def HelmbergerSignal(self, dt, npts, k=16.8, B=2.04):
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        KT=k*time
        self.data=1.-np.exp( -KT )*( 1.+KT+np.power(KT,2)/2.-B*np.power(KT,3) )
        return
    
    def StepSignal(self, dt, npts):
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        self.data=np.ones(time.size)
        return
    
    def RickerWavelet(self, dt, npts, fpeak):
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        a=1./(fpeak*math.pi)
        self.data=scipy.signal.ricker(npts, a=a)
        return
    
    def GaussianSignal(self, dt, npts, fc, t0=None):
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        w=2*np.pi*fc
        sigma=1/w
        if t0==None:
            Nshift2=np.int(6./w/dt) + 1
            print w
            t0=Nshift2 *dt
        else:
            Nshift2=np.int(t0/dt) + 1
            t0=Nshift2 *dt
        if npts % 2 ==0:
            tempSig=scipy.signal.gaussian(M=npts+1, std=sigma)
            Nshift1=npts/2+1
        else:
            tempSig=scipy.signal.gaussian(M=npts, std=sigma)
            Nshift1=(npts+1)/2
        tempSig=np.roll(tempSig, Nshift1)
        tempSig=np.roll(tempSig, -Nshift2)
        tempSig[Nshift1:]=0.
        self.data=tempSig[:npts]
        return

    def RickerIntSignal(self, dt, npts, fc, t0=None):
        time=np.arange(npts)*dt
        self.stats.npts=npts
        self.stats.delta=dt
        w=2*np.pi*fc
        sigma=1/w
        if t0==None:
            Nshift2=np.int(6./w/dt) + 1
            print w
            t0=Nshift2 *dt
        else:
            Nshift2=np.int(t0/dt) + 1
            t0=Nshift2 *dt
        if npts % 2 ==0:
            tempSig=scipy.signal.gaussian(M=npts+1, std=sigma)
            Nshift1=npts/2+1
        else:
            tempSig=scipy.signal.gaussian(M=npts, std=sigma)
            Nshift1=(npts+1)/2
        tempSig=np.diff(tempSig)
        tempSig=np.roll(tempSig, Nshift1)
        tempSig=np.roll(tempSig, -Nshift2)
        tempSig[Nshift1:]=0.
        self.data=tempSig[:npts]
        return 
    
    def dofft(self, diff=True):
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
    
class ses3dCatalog(obspy.core.event.Catalog):
    
    def add_event(self, longitude, latitude, depth, event_type='earthquake', focalmechanism=None):
        origin=obspy.core.event.origin.Origin(longitude=longitude, latitude=latitude, depth=depth)
        event=obspy.core.event.event.Event(origins=[origin], event_type=event_type, focal_mechanisms=[focalmechanism])
        self.append(event)
        return
    
    
    def add_explosion(self, longitude, latitude, depth, m0):
        tensor = obspy.core.event.source.Tensor(m_rr=m0, m_tt=m0, m_pp=m0,
                                                 m_tp=0., m_rt=0., m_rp=0.)
        moment_tensor = obspy.core.event.source.MomentTensor(tensor=tensor, scalar_moment=m0)
        FocalMech = obspy.core.event.source.FocalMechanism(moment_tensor=moment_tensor)
        self.add_event(longitude=longitude, latitude=latitude, depth=depth, event_type='explosion', focalmechanism=FocalMech)
        return
    
    def add_earthquake(self, longitude, latitude, depth, m_rr, m_tt, m_pp, m_tp, m_rt, m_rp):
        tensor = obspy.core.event.source.Tensor(m_rr=m_rr, m_tt=m_tt, m_pp=m_pp,
                                                 m_tp=m_tp, m_rt=m_rt, m_rp=m_rp)
        moment_tensor = obspy.core.event.source.MomentTensor(tensor=tensor)
        FocalMech = obspy.core.event.source.FocalMechanism(moment_tensor=moment_tensor)
        self.add_event(longitude=longitude, latitude=latitude, depth=depth, focalmechanism=FocalMech)
        return
    
    def write4ses3d(self, outdir):
        eventlst=outdir+'/event_list'
        
        