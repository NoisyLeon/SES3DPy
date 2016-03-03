# import SES3DPy#!/usr/bin/env python
# import matplotlib.pyplot as plt
# import matplotlib.pylab as plb
# minlat=25.;
# maxlat=52.;
# minlon=90.;
# maxlon=143.;
# # stafile='/projects/life9360/EA_MODEL/EA_grid_point.lst';
# datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_SAC_EA_10sec_1km_001/NKNT';
# outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_DISP_EA_10sec_1km_001'
# staLst=SES3DPy.StaLst();
# # staLst.ReadStaList(stafile);
# staLst.GenerateReceiverLst(minlat=minlat, maxlat=maxlat,\
#     minlon=minlon, maxlon=maxlon, dlat=0.5, dlon=0.5, factor=10, net='EA', PRX='');
# # NSLst=staLst.GetGridStaLst();
# ses3dST=SES3DPy.ses3dStream();
# ses3dST.Getses3dsynDIST(datadir=datadir, SLst=staLst, mindist=1075, maxdist=1125, channel='BXZ');
# title='1050 km < dist < 1150 km'
# fig=plb.figure(num=1, figsize=(8.,12.), facecolor='w', edgecolor='k');
# ses3dST.PlotDISTStreams( title=title)
# degree_sign= u'\N{DEGREE SIGN}'
# plt.ylabel('Azimuth('+degree_sign+')', fontsize=20)
# plt.xlabel('Time(sec)', fontsize=20)
# plt.title('Synthetic Seismogram for 1075 km < D < 1125 km', fontsize=35)
# plt.ylim(120,335)
# plt.xlim(200,600)
# plt.show()


import GeoPoint as GP#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import GeoPolygon as GeoPoly
mapfile='/projects/life9360/EA_MODEL/EA_grid_point.lst';
datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_SAC_EA_10sec_1km_001/NKNT';
outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_DISP_EA_10sec_1km_001'
Ampfname='/projects/life9360/code/ses3dPy/10sec/NKNT_Amplitude.lst'
minlat=25.;
maxlat=52.;
minlon=90.;
maxlon=143.;
evlo=129.029;
evla=41.306;
GeoLst=GP.GeoMap()
# GeoLst.ReadGeoMapLst(mapfile)
GeoLst.GetGridGeoMap(maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat)
# TrimGeoLst=GeoLst.Trim(maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat)
# myGeoLst=TrimGeoLst.GetINTGeoMap();
# GeoLst.StationDistDistribution(evlo, evla, infname=Ampfname, mindist=1675., maxdist=1725., \
#         maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat, mapflag='regional_merc')
mygeopolygons=GeoPoly.GeoPolygonLst();
mygeopolygons.ReadGeoPolygonLst('basin1');
GeoLst.StationSpecialDistribution(evlo, evla, minazi=155, maxazi=160, mindist=0, maxdist=1400., \
        maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat, mapflag='regional_merc', \
        infname=Ampfname, geopolygons=mygeopolygons)