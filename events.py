
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