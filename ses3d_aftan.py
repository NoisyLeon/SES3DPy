import ses3d_noisepy as npy
import numpy as np
stafile='/projects/life9360/EA_MODEL/EA_grid_point.lst';
datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_SAC_EA_10sec_1km_001';
outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_DISP_EA_10sec_1km_001'
staLst=npy.StaLst();
staLst.ReadStaList(stafile);
NSLst=staLst.GetGridStaLst();
evlo=129.029;
evla=41.306;
Mw=4.1;
evdp=1.0;
event=npy.StaInfo(stacode='NKNT', lon=evlo, lat=evla);
event.datatype='ses3d';
event.type='event';
# NSLst.append(event)
### Generate Predicted Phase V Dispersion
# event.GeneratePrePhaseDISP(NSLst, outdir=outdir+'/PREPHASE');

### aftan analysis
# Prepare database
predir=outdir+'/PREPHASE_R';
CHAN=[ ['BXZ'] ]
inftan=npy.InputFtanParam() 
inftan.setInParam(tmin=3.0, tmax=40.0);  # Set Input aftan parameters
# staLst.MakeDirs(outdir=outdir, dirtout='DISP'); ### Important!
StapairLst=npy.StaPairLst(); # 
StapairLst.GenerateEventStaPairLst([event], NSLst, chanAll=CHAN); # Generate station pair list from station list
StapairLst.set_PRE_PHASE(predir=predir);
# StapairLst.aftanSNR(datadir, outdir, tin='', tout='', dirtin='', dirtout='DISP', inftan=inftan)

# Now do aftan
# L=len(StapairLst);
# Lm=int(L/20)
# for i in np.arange(Lm):
#     print i
#     tempstapair=StapairLst[i*20:(i+1)*20];
#     tempstapair.set_PRE_PHASE(predir=predir);
#     tempstapair.aftanSNRParallel(datadir=datadir, outdir=outdir, tin='COR', tout='COR', dirtin='COR', dirtout='DISP', inftan=inftan);
# tempstapair=StapairLst[Lm*20:];
# tempstapair.set_PRE_PHASE(predir=predir);
# tempstapair.aftanSNRParallel(datadir=datadir, outdir=outdir, tin='COR', tout='COR', dirtin='COR', dirtout='DISP', inftan=inftan);
# StapairLst.aftanSNRParallel(datadir=datadir, outdir=outdir, tin='COR', tout='COR', dirtin='COR', dirtout='DISP', inftan=inftan);

# ## Eikonal Tomography
datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_DISP_EA_10sec_1km_001';
outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_TravelTime_MAP'
per_array=np.arange((32-6)/2+1)*2.+6.
minlat=30.
maxlat=50.
minlon=110.
maxlon=140.
# event.GetTravelTimeFile(NSLst, 10, datadir, outdir, dirtin='DISP', \
#             minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, tin='', dx=0.2, filetype='phase', chpair=['BXZ'] )
dx=0.2;
npts_x=int((maxlon-minlon)/dx)+1
npts_y=int((maxlat-minlat)/dx)+1
# event.CheckTravelTimeCurvature(10, outdir, minlon, npts_x, minlat, npts_y, dx=0.2, filetype='phase');
# event.TravelTime2Slowness(datadir=outdir, outdir=outdir, per=10,\
#     minlon=minlon, npts_x=npts_x, minlat=minlat, npts_y=npts_y, dx=0.2, cdist=None, filetype='phase' )
# 
# event.GetTravelTimeFile(NSLst, 10, datadir, outdir, dirtin='DISP', \
#             minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, tin='', dx=0.2, filetype='group', chpair=['BXZ'] )
# event.CheckTravelTimeCurvature(10, outdir, minlon, npts_x, minlat, npts_y, dx=0.2, filetype='group');
# event.TravelTime2Slowness(datadir=outdir, outdir=outdir, per=10,\
#     minlon=minlon, npts_x=npts_x, minlat=minlat, npts_y=npts_y, dx=0.2, cdist=None, filetype='group' )

## Get Azimuth distribution of ampplitude and surface wave magnitude
# event.GetAmpAzi(NSLst, 10, datadir, outdir, dirtin='DISP', \
#             minazi=155, maxazi=160, tin='', chpair=['BXZ'] )
event.GetAmpDist(NSLst, 10, datadir, outdir, dirtin='DISP', \
            mindist=1550, maxdist=1600, tin='', chpair=['BXZ'] )
# event.GetAmplitudeFile(NSLst, 10, datadir, outdir, dirtin='DISP', \
#             minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, tin='', dx=0.2, chpair=['BXZ'] )
# event.CheckTravelTimeCurvature(10, outdir, minlon, npts_x, minlat, npts_y, dx=0.2, filetype='group');
# event.TravelTime2Slowness(datadir=outdir, outdir=outdir, per=10,\
#     minlon=minlon, npts_x=npts_x, minlat=minlat, npts_y=npts_y, dx=0.2, cdist=None, filetype='group' )
# staLst.CheckTravelTimeCurvatureParallel(perlst=per_array, outdir=outdir+'/Eikonal_out', minlon=-125, maxlon=-105, minlat=31, maxlat=50, dx=0.2, filetype='phase');
# 
# 
# datadir='/lustre/janus_scratch/life9360/ancc-1.0-0/Eikonal_out'
# outdir='/lustre/janus_scratch/life9360/ancc-1.0-0/Eikonal_out'
# staLst.TravelTime2SlownessParallel(datadir=outdir, outdir=outdir, perlst=per_array, minlon=-125, maxlon=-105, minlat=31, maxlat=50, filetype='phase')
# npy.Slowness2IsoAniMap(stafile=stafile, perlst=per_array, datadir=outdir, outdir=outdir, minlon=-125, maxlon=-105, minlat=31, maxlat=50,\
#     dx=0.2, pflag=1, crifactor=12, minazi=-180, maxazi=180, N_bin=20)

