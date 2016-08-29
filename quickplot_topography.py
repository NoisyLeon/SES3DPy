import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from matplotlib.mlab import griddata
import matplotlib.cm
from matplotlib.ticker import FuncFormatter
import numexpr as npr
import glob
from functools import partial
import multiprocessing as mp
import obspy.core.util.geodetics as obsGeo
import math
from netCDF4 import Dataset
import pycpt
import GeoPolygon
import obspy.geodetics.base

def PlotTopography(evlo, evla, maxlat, minlat, maxlon, minlon, projection='lambert', res='i', geopolygons=None):
    """
    Read Sation List from a txt file
    stacode longitude latidute network
    """

    lon_min=minlon
    lat_min=minlat
    lon_max=maxlon
    lat_max=maxlat
    lat_centre = (lat_max+lat_min)/2.0
    lon_centre = (lon_max+lon_min)/2.0
    fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
    # ax=plt.subplot(111)
    # m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, \
    #     rsphere=(6378137.00,6356752.3142), resolution='c', projection='merc')
    if projection=='global':
        m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
        m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
        m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
    elif projection=='regional_ortho':
        m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
        m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
            llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/mapfactor)
        # labels = [left,right,top,bottom]
        m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,0])
        m.drawmeridians(np.arange(-170.0,170.0,10.0))	
    elif projection=='regional_merc':
        m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
        m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
        m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
    elif projection=='lambert':
         distEW, az, baz=obspy.geodetics.base.gps2dist_azimuth(lat_min, lon_min,
                             lat_min, lon_max) # distance is in m
         distNS, az, baz=obspy.geodetics.base.gps2dist_azimuth(lat_min, lon_min,
                            lat_max+1.7, lon_min) # distance is in m
         m = Basemap(width=distEW, height=distNS,
         rsphere=(6378137.00,6356752.3142),\
         resolution='l', projection='lcc',\
         lat_1=lat_min, lat_2=lat_max, lat_0=lat_centre+1.2, lon_0=lon_centre)
         m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=2, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
         m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=2, dashes=[2,2], labels=[0,0,1,1], fontsize=15)
    # lon = LonLst
    # lat = LatLst
    # x,y = m(lon, lat)
    # m.plot(x, y, 'r^', markersize=2)
    m.drawcoastlines()
    evx, evy=m(evlo, evla)
    m.plot(evx, evy, 'r*', markersize=15)
    # InArr=np.loadtxt('/projects/life9360/station_map/ETOPO1_Ice_g_int.xyz')
    # etopo=InArr[:,2];
    # lons=InArr[:,0];
    # lats =InArr[:,1];
    # lats=lats[lats>0];
    # lons=lons[lats>0];
    # etopo=etopo[lats>0]
    # print 'End of Reading Data'
    # ddir='/projects/life9360/software/basemap-1.0.7/examples';
    # etopo = np.loadtxt(ddir+'/etopo20data.gz')
    # lons  = np.loadtxt(ddir+'/etopo20lons.gz')
    # lats  = np.loadtxt(ddir+'/etopo20lats.gz')
    # url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc'
    blon=np.arange(100)*(maxlon-minlon)/100.+minlon
    blat=np.arange(100)*(maxlat-minlat)/100.+minlat
    Blon=blon;
    Blat=np.ones(Blon.size)*minlat;
    x,y = m(Blon, Blat)
    m.plot(x, y, 'b--', lw=4)
    
    Blon=blon;
    Blat=np.ones(Blon.size)*maxlat;
    x,y = m(Blon, Blat)
    m.plot(x, y, 'b--', lw=4)
    
    Blon=np.ones(Blon.size)*minlon;
    Blat=blat;
    x,y = m(Blon, Blat)
    m.plot(x, y, 'b--', lw=4)
    
    Blon=np.ones(Blon.size)*maxlon;
    Blat=blat;
    x,y = m(Blon, Blat)
    m.plot(x, y, 'b--', lw=4)
    
    # mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
    # mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
    # etopodata = Dataset('/projects/life9360/station_map/grd_dir/etopo20.nc')
    # # topoin = etopodata.variables['z'][:]
    # # lons = etopodata.variables['lon'][:]
    # # lats = etopodata.variables['lat'][:]
    # etopo = etopodata.variables['ROSE'][:]
    # lons = etopodata.variables['ETOPO20X1_1081'][:]
    # lats = etopodata.variables['ETOPO20Y'][:]
    # # etopo=topoin[((lats>20)*(lats<85)), :];
    # # etopo=etopo[:, (lons>85)*(lons<180)];
    # # lats=lats[(lats>20)*(lats<85)];
    # # lons=lons[(lons>85)*(lons<180)];
    # 
    # x, y = m(*np.meshgrid(lons,lats))
    # mycm2.set_over('w',0)
    # m.pcolormesh(x, y, etopo, shading='gouraud', cmap=mycm1, vmin=0, vmax=8000)
    # m.pcolormesh(x, y, etopo, shading='gouraud', cmap=mycm2, vmin=-11000, vmax=-0.5)
    m.etopo()
    try:
        geopolygons.PlotPolygon(inbasemap=m)
    except:
        pass
    # draw parallels
    plt.show()
    return

if __name__ == '__main__':
    basins=GeoPolygon.GeoPolygonLst()
    basins.ReadGeoPolygonLst('basin1')
    evlo=129.0
    # evlo=129.029
    evla=41.306
    minlat=22.
    maxlat=52.
    minlon=85.
    maxlon=133.
    PlotTopography(evlo, evla, maxlat, minlat, maxlon, minlon, projection='lambert', res='i', geopolygons=basins)