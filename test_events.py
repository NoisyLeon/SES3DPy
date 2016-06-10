import events

dt=0.05
num_timpstep=100


fmin=1./100.;
fmax=1./10.
STF=events.STF()
STF.RickerIntSignal(dt=dt, npts=num_timpstep, fc=0.1)
STF.plot()
# STF.StepSignal(dt=dt, npts=num_timpstep)
# STF.filter('highpass', freq=fmin, corners=4, zerophase=False)
# STF.filter('lowpass', freq=fmax, corners=4, zerophase=False)
STF.plotfreq()