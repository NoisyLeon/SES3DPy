import SES3DPy

outdir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/INPUT'

evlo=15.;
evla=2.;
Mw=4.1;
evdp=1.0;

gen=SES3DPy.SES3DInputGenerator();

#SES3D configuration
num_timpstep=10000;

minlat=0.
maxlat=4.
minlon=10.
maxlon=18.

mindep=0.;
maxdep=200.;
dt=0.05;

nx_global=64;
ny_global=120;
nz_global=30;


px=8;
py=12;
pz=5;

gen.AddExplosion(lon=evlo, lat=evla, depth=evdp, mag=Mw);

gen.SetConfig(num_timpstep=num_timpstep, dt=dt, nx_global=nx_global, ny_global=ny_global, nz_global=nz_global,\
    px=px, py=py, pz=pz, minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon, mindep=mindep, maxdep=maxdep)
# gen.make_stf(dt=dt, fmin=0.1, fmax=0.1, nt=num_timpstep, plotflag=True);
STF=SES3DPy.SourceTimeFunc();
STF.StepSignal(dt=dt, npts=num_timpstep)
gen.get_stf(stf=STF, fmin=0.01, fmax=0.1, plotflag=True)
# gen.GenerateReceiverLst(minlat, maxlat, minlon, maxlon, dlat=0.1, dlon=0.1, PRX='LF')
# gen.WriteSES3D(outdir=outdir);

