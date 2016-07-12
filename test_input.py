import input_generator
import events
import stations


evlo=129.029
evla=41.306
Mw=4.1
evdp=1.0

# 
# outdir='.'
# 
# #SES3D configuration
# num_timpstep=36000;
# 
# minlat=25.
# maxlat=52.
# minlon=90.
# maxlon=143.
# 
# zmin=0.;
# zmax=400.;
# dt=0.05;
# 
# nx_global=320;
# ny_global=564;
# nz_global=42;
# 
# px=10;
# py=12;
# pz=3;
# 
# inGen=input_generator.InputFileGenerator()
# inGen.set_config(num_timpstep=num_timpstep, dt=dt, nx_global=nx_global, ny_global=ny_global, nz_global=nz_global,\
#     px=px, py=py, pz=pz, minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon, zmin=zmin, zmax=zmax)
# inGen.add_explosion(longitude=45.2, latitude=32.1, depth=1., m0=1e16)
# STF=events.STF()
# STF.GaussianSignal(dt=dt, npts=num_timpstep, fc= 0.1)
# # inGen.get_stf(stf=STF, fmin=0.01, fmax=0.05)
# inGen.get_stf(stf=STF)
# # inGen.add_stations(stafile)
# # inGen.write(outdir=outdir)
