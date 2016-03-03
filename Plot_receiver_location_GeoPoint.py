import GeoPoint as GP#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.pylab as plb

minlat=25.;
maxlat=52.;
minlon=90.;
maxlon=143.;
evlo=129.029;
evla=41.306;
GeoLst=GP.GeoMap()
GeoLst.GetGridGeoMap(maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat)
# TrimGeoLst=GeoLst.Trim(maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat)
# myGeoLst=TrimGeoLst.GetINTGeoMap();
GeoLst.StationDistribution(evlo=evlo, evla=evla, maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat,\
        mapfactor=2.5);
# ses3dST=SES3DPy.ses3dStream();
# ses3dST.Getses3dsynDIST(datadir=datadir, SLst=NSLst, mindist=1050., maxdist=1150., channel='BXZ');
# title='1050 km < dist < 1150 km'
# fig=plb.figure(num=1, figsize=(8.,12.), facecolor='w', edgecolor='k');
# ses3dST.PlotDISTStreams( title=title)
# plt.show()

