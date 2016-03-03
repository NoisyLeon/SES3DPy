# -*- coding: utf-8 -*-
# import SES3DPy#!/usr/bin/env python
# import matplotlib.pyplot as plt
# import matplotlib.pylab as plb
# stafile='/projects/life9360/EA_MODEL/EA_grid_point.lst';
# datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_SAC_EA_10sec_1km_001/NKNT';
# outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_DISP_EA_10sec_1km_001'
# staLst=SES3DPy.StaLst();
# staLst.ReadStaList(stafile);
# NSLst=staLst.GetGridStaLst();
# ses3dST=SES3DPy.ses3dStream();
# ses3dST.Getses3dsynAzi(datadir=datadir, SLst=NSLst, minazi=155, maxazi=160, channel='BXZ');
# title='270 deg < azi < 275 deg'
# fig=plb.figure(num=1, figsize=(8.,12.), facecolor='w', edgecolor='k');
# ses3dST.PlotAziStreams( title=title)
# plt.ylabel('Distance(km)', fontsize=20)
# plt.xlabel('Time(sec)', fontsize=20)
# degree_sign= u'\N{DEGREE SIGN}'
# plt.title('Synthetic Seismogram for 155'+degree_sign+' < azimuth < 160'+degree_sign, fontsize=35)
# # plt.ylim(120,330)
# # plt.show()
# plt.show()#!/usr/bin/env python


import GeoPoint as GP#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import GeoPolygon as GeoPoly
mapfile='/projects/life9360/EA_MODEL/EA_grid_point.lst';
datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_SAC_EA_10sec_1km_001/NKNT';
outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT_DISP_EA_10sec_1km_001'
Ampfname='/projects/life9360/code/ses3dPy/10sec/NKNT_Amplitude.lst'
diffa_t='/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_TravelTime_MAP/10sec/diffa.map'
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
# GeoLst.StationAziDistribution(evlo, evla, minazi=235, maxazi=236, \
#         maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat, mapflag='regional_merc', infname=diffa_t)
mygeopolygons=GeoPoly.GeoPolygonLst();
mygeopolygons.ReadGeoPolygonLst('basin1');
GeoLst.StationSpecialDistribution(evlo, evla, minazi=155, maxazi=160, mindist=0, maxdist=1400., \
        maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat, mapflag='regional_merc',\
        infname=Ampfname, geopolygons=mygeopolygons)