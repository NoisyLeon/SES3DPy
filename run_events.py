import events

dt=0.05
num_timpstep=36001
# 
# 
fmin=1./100.;
fmax=1./10.
STF=events.STF()
STF.GaussianSignal(dt=dt, npts=num_timpstep, fc=0.1)
STF.plot()
# STF.StepSignal(dt=dt, npts=num_timpstep)
# STF.filter('highpass', freq=fmin, corners=4, zerophase=False)
# STF.filter('lowpass', freq=fmax, corners=4, zerophase=False)
STF.plotfreq()

# catalog=events.ses3dCatalog()
# catalog.add_explosion(longitude=45.2, latitude=32.1, depth=1., m0=1e16)
# ef=catalog.write(outdir='./', nt=4000, dt=0.05, output_folder='./OUTPUT', ssamp=100, outdisp=1)
