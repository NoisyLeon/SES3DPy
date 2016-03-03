import SES3DPy

# stafile='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/station.lst_ses3d';
stafile='/projects/life9360/EA_MODEL/EA_grid_point.lst';
outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT';
outdir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/INPUT'

evlo=129.029;
evla=41.306;
Mw=4.1;
evdp=1.0;

staLst=SES3DPy.StaLst();
staLst.ReadStaList(stafile);
gen=SES3DPy.SES3DInputGenerator();

#SES3D configuration
num_timpstep=20000;
# minlat=15.
# maxlat=50.
# minlon=70.
# maxlon=150.

minlat=30.
maxlat=50.
minlon=110.
maxlon=140.
# minlat=19.;
# maxlat=63.;
# minlon=102.0;
# maxlon=156.0;
mindep=0.;
maxdep=500.;
dt=0.04;
# nx_global=336;
# ny_global=504;
# nz_global=50;

nx_global=288;
ny_global=396;
nz_global=70;


px=8;
py=12;
pz=5;

# nx_global=36;
# ny_global=60;
# nz_global=20;
# px=3;
# py=4;
# pz=2;
NSLst=staLst.GetGridStaLst();
# gen.StaLst2Generator(NSLst, minlon=minlon, maxlon=maxlon, minlat=minlat , maxlat=maxlat);
gen.AddExplosion(lon=evlo, lat=evla, depth=evdp, mag=Mw);

gen.SetConfig(num_timpstep=num_timpstep, dt=dt, nx_global=nx_global, ny_global=ny_global, nz_global=nz_global,\
    px=px, py=py, pz=pz, minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon, mindep=mindep, maxdep=maxdep)
gen.make_stf(dt=dt, fmin=0.11, fmax=0.11, nt=num_timpstep, plotflag=True);
# gen.WriteSES3D(outdir=outdir);