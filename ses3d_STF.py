import SES3DPy
import matplotlib.pyplot as plt
import numpy as np
import obspy
dt=0.1;
npts=10000;
tauR=1
f='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT_SAC_TEST_PML_10sec_001/LF.LF160S1710..BXZ.SAC'


# STF=obspy.read(f)[0];
# STF=SES3DPy.SourceTimeFunc(STF.data, STF.stats);
# STF.filter('bandpass', freqmin=0.01,freqmax=0.1, corners=4, zerophase=False)
# # STF1=STF.copy();
# # STF1.decimate(2);
# STF.dofft();
# STF.plotfreq();
# # STF1.dofft();
# # STF1.plotfreq();


STF=SES3DPy.SourceTimeFunc();
# STF.BruneSignal(dt=dt, npts=npts)
# STF.RickerWavelet(dt=dt, npts=npts, fpeak=0.1)
# STF.HaskellSignal(dt=dt, npts=npts)
STF.vonSeggernSignal(dt=dt, npts=npts)
# STF.StepSignal(dt=dt, npts=npts)
# STF.filter('lowpass', freq=0.2)
# STF.filter('highpass', freq=0.01)
# STF.filter('bandstop', freqmin=0.01,freqmax=0.1, corners=5, zerophase=False)
STF.filter('highpass', freq=0.02, corners=4, zerophase=False)
STF.filter('lowpass', freq=0.1, corners=5, zerophase=False)
ax=plt.subplot(211);
STF.dofft();
STF.plotfreq();

dt=0.05;
npts=20000;
STF1=STF.copy();
STF1.vonSeggernSignal(dt=dt, npts=npts)
# STF1.BruneSignal(dt=dt, npts=npts)
# STF1.RickerWavelet(dt=dt, npts=npts, fpeak=0.1)
# STF1.StepSignal(dt=dt, npts=npts)
# STF.filter('lowpass', freq=0.2)
# STF.filter('highpass', freq=0.01)
# STF1.filter('bandstop', freqmin=0.01,freqmax=0.1, corners=5, zerophase=False)
STF1.filter('highpass', freq=0.02, corners=4, zerophase=False)
STF1.filter('lowpass', freq=0.1, corners=5, zerophase=False)

STF1.dofft();
STF1.plotfreq();
# STF.vonSeggernSignal(dt=dt, npts=npts)
# STF1.HaskellSignal(dt=dt, npts=npts)

ax=plt.subplot(212);
ax.plot(np.arange(STF.stats.npts)*STF.stats.delta, STF.data, 'b-', lw=2);
ax.plot(np.arange(STF1.stats.npts)*STF1.stats.delta, STF1.data, 'r.-', lw=2);
# ax.plot(np.arange(STF.stats.npts)*STF.stats.delta, STF.mydata, 'g.-', lw=2);
plt.xlim(0, 200)
