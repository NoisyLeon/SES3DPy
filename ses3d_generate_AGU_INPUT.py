import SES3DPy
import matplotlib.pyplot as plt

outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT'
stafile='/projects/life9360/EA_MODEL/EA_grid_point.lst';

evlo=129.029;
evla=41.306;
Mw=4.1;
evdp=1.0;

#SES3D configuration
num_timpstep=36000;

# minlat=30.
# maxlat=50.
# minlon=110.
# maxlon=140.

minlat=25.
maxlat=52.
minlon=90.
maxlon=143.

mindep=0.;
maxdep=400.;
dt=0.05;

nx_global=320;
ny_global=564;
nz_global=42;

px=10;
py=12;
pz=3;
##############
# Source Time Function
##############
fmin=1./100.;
fmax=1./10.
STF=SES3DPy.SourceTimeFunc();
STF.StepSignal(dt=dt, npts=num_timpstep)
STF.filter('highpass', freq=fmin, corners=4, zerophase=False)
STF.filter('lowpass', freq=fmax, corners=4, zerophase=False)
# STF.filter('bandstop', freqmin=0.01,freqmax=0.1, corners=4, zerophase=False)
###################

#
# staLst.ReadStaList(stafile);
gen=SES3DPy.SES3DInputGenerator();
# NSLst=staLst.GetGridStaLst();

gen.GetRelax(datadir=outdir)
# # gen.StaLst2Generator(NSLst, minlon=minlon, maxlon=maxlon, minlat=minlat , maxlat=maxlat);
gen.AddExplosion(lon=evlo, lat=evla, depth=evdp, mag=Mw);

gen.SetConfig(num_timpstep=num_timpstep, dt=dt, nx_global=nx_global, ny_global=ny_global, nz_global=nz_global,\
    px=px, py=py, pz=pz, minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon, mindep=mindep, maxdep=maxdep)
gen.CheckMinWavelengthCondition(NGLL=4., fmax=1.0/10.0, Vmin=1.2)
gen.get_stf(stf=STF, fmin=fmin, fmax=fmax, plotflag=True)
# gen.WriteSES3D(outdir=outdir);
# Receiver List
staLst=SES3DPy.StaLst();
staLst.GenerateReceiverLst(minlat=minlat, maxlat=maxlat,\
    minlon=minlon, maxlon=maxlon, dlat=0.25, dlon=0.25, net='EA', PRX='');
# staLst.WriteRecfile(outdir=outdir)