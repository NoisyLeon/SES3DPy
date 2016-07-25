import input_generator
import events
import stations
import Qmodel

outdir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT'
#########################
# Qmodel

# Qmodel=Qmodel.Qmodel( fmin=1.0/100., fmax=1/5.)
# Qmodel.Qdiscrete()
# 
# # D=np.array([1.684, 0.838, 1.357]);
# # tau_s=np.array([3.2, 17.692, 74.504]);
# # Qmodel.PlotQdiscrete( D=D, tau_s=tau_s );
# Qmodel.PlotQdiscrete()
# Qmodel.write(outdir=outdir)
#########################

# #################
# stations
SLst=stations.StaLst()
SLst.HomoStaLst(minlat = 22.5, Nlat = 116, minlon=85.5, Nlon = 188, dlat=0.25, dlon=0.25, net='SES')
# SLst.write(outdir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT')
#################
# events
evlo=129.0
# evlo=129.029
evla=41.306
Mw=4.1
evdp=1.0

dt=0.05
num_timpstep=60000
# fmin=1./100.;
# fmax=1./10.
STF=events.STF()
STF.RickerIntSignal(dt=dt, npts=num_timpstep, fc=0.1)
# STF.plot()
# STF.StepSignal(dt=dt, npts=num_timpstep)
# STF.filter('highpass', freq=fmin, corners=4, zerophase=False)
# STF.filter('lowpass', freq=fmax, corners=4, zerophase=False)
# STF.plotfreq()

##################
outdir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT'
#SES3D configuration
num_timpstep=60000
minlat=22.
maxlat=52.
minlon=85.
maxlon=133.

zmin=0.
zmax=200.

nx_global=420
ny_global=624
nz_global=27

# nx_global=670
# ny_global=996
# nz_global=42

px=10
py=12
pz=3

inGen=input_generator.InputFileGenerator()
inGen.set_config(num_timpstep=num_timpstep, dt=dt, nx_global=nx_global, ny_global=ny_global, nz_global=nz_global,\
    px=px, py=py, pz=pz, minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon, zmin=zmin, zmax=zmax)
inGen.add_explosion(longitude=evlo, latitude=evla, depth=1., m0=1.45e15)
# STF=events.STF()
# STF.GaussianSignal(dt=dt, npts=num_timpstep, fc= 0.1)
# inGen.get_stf(stf=STF, fmin=0.01, fmax=0.1, vmin=1.)
inGen.get_stf(stf=STF, vmin=2.0)
inGen.add_stations(SLst)
inGen.write(outdir=outdir)
