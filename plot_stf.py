import matplotlib.pyplot as plt
import events



dt=0.05
num_timpstep=50000
fmin=1./20.
fmax=1./10.
STF=events.STF()
# STF.StepSignal(dt=dt, npts=num_timpstep)
STF.RickerIntSignal(dt=dt, npts=num_timpstep, fc=0.1)
# STF.filter('bandpass',freqmin=fmin, freqmax=fmax )
STF.data=STF.data*20/83.44
fig=plt.figure(figsize=(10, 20))
stime=STF.stats.starttime
ax1=plt.subplot(211)
STF.plottime(te=100.)
ax1.xaxis.set_label_coords(0.5, -0.05)
# STF.plot(starttime=stime, endtime=stime+200., type='relative')
ax2=plt.subplot(212)
STF.plotperiod()
ax2.xaxis.set_label_coords(0.5, -0.05)
plt.show()
