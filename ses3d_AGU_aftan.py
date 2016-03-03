import ses3d_noisepy as npy
import numpy as np
datadir='/lustre/janus_scratch/life9360/EA_postprocessing_AGU';
outdir='/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_DISP'
minlat=25.
maxlat=52.
minlon=90.
maxlon=143.
staLst=npy.StaLst();
staLst.GenerateReceiverLst(minlat=minlat, maxlat=maxlat,\
    minlon=minlon, maxlon=maxlon, dlat=0.5, dlon=0.5, factor=10, net='EA', PRX='');
evlo=129.029;
evla=41.306;
Mw=4.1;
evdp=1.0;
event=npy.StaInfo(stacode='NKNT', lon=evlo, lat=evla);
event.datatype='ses3d';
event.type='event';

######################################
#Generate Predicted Phase V Dispersion
######################################
# event.GeneratePrePhaseDISP(staLst, outdir=outdir+'/PREPHASE');

################################
#aftan analysis
################################
# Prepare database
predir=outdir+'/PREPHASE_R';
CHAN=[ ['BXZ'] ]
inftan=npy.InputFtanParam() 
inftan.setInParam(vmin=1.5, tmin=5.0, tmax=70.0);  # Set Input aftan parameters
# staLst.MakeDirs(outdir=outdir, dirtout='DISP'); ### Important!
StapairLst=npy.StaPairLst(); # 
StapairLst.GenerateEventStaPairLst([event], staLst, chanAll=CHAN); # Generate station pair list from station list
StapairLst.set_PRE_PHASE(predir=predir);
# StapairLst.aftanSNR(datadir, outdir, tin='', tout='', dirtin='', dirtout='DISP', inftan=inftan)
# StapairLst.aftanSNRParallel(datadir, outdir, tin='', tout='', dirtin='', dirtout='DISP', inftan=inftan);

################################
#Eikonal Tomography
################################
datadir='/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_DISP';
outdir='/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_TravelTime_MAP';
# per_array=np.arange((32-6)/2+1)*2.+6.
per_array=np.arange((50-10)/2+1)*2.+10.;
# per_array=np.array([10.])
minlat=25.;
maxlat=52.;
minlon=90.;
maxlon=143.;
staLst=npy.StaLst();
staLst.GenerateReceiverLst(minlat=minlat, maxlat=maxlat,\
    minlon=minlon, maxlon=maxlon, dlat=0.5, dlon=0.5, factor=100, net='EA', PRX='');

eventLst=npy.StaLst();
eventLst.append(event);
# eventLst.GetTravelTimeFile(staLst, per_array, datadir, outdir, dirtin='DISP', \
#     minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, tin='', dx=0.2, filetype='phase', chpair=['BXZ'] );
# dx=0.2;
# npts_x=int((maxlon-minlon)/dx)+1;
# npts_y=int((maxlat-minlat)/dx)+1;
# eventLst.CheckTravelTimeCurvature(perlst=per_array, outdir=outdir, \
#     minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat);
# eventLst.TravelTime2Slowness(datadir=outdir, outdir=outdir, perlst=per_array,\
#         minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat);
# eventLst.GetAmplitudeFile(staLst, per_array, datadir, outdir, dirtin='DISP', \
#         minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, tin='', dx=0.2, chpair=['BXZ'] )

###################################################################
#Get Azimuth distribution of ampplitude and surface wave magnitude
###################################################################
myazi=235
event.GetAmpAzi(staLst, 10, datadir, outdir, dirtin='DISP', \
            minazi=myazi, maxazi=myazi+1, tin='', chpair=['BXZ'] )

# event.GetAmpDist(staLst, 10, datadir, outdir, dirtin='DISP', \
#             mindist=1675, maxdist=1725, tin='', chpair=['BXZ'] )
# # event.GetAmplitudeFile(NSLst, 10, datadir, outdir, dirtin='DISP', \
# #             minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, tin='', dx=0.2, chpair=['BXZ'] )
# event.CheckTravelTimeCurvature(10, outdir, minlon, npts_x, minlat, npts_y, dx=0.2, filetype='group');
# event.TravelTime2Slowness(datadir=outdir, outdir=outdir, per=10,\
#     minlon=minlon, npts_x=npts_x, minlat=minlat, npts_y=npts_y, dx=0.2, cdist=None, filetype='group' )
# staLst.CheckTravelTimeCurvatureParallel(perlst=per_array, outdir=outdir+'/Eikonal_out', minlon=-125, maxlon=-105, minlat=31, maxlat=50, dx=0.2, filetype='phase');




