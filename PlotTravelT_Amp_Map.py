import numpy as np
import matplotlib.pyplot as plt
import os,sys;
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
import matplotlib.cm
from matplotlib.ticker import FuncFormatter
# sys.path.append('/projects/life9360/.local/quickkey')
# import colormaps as cm
# foo.MyClass()

fname=sys.argv[1]
dlon=float(sys.argv[2])
dlat=float(sys.argv[3])
# my_colormap=cm.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0],\
#                 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], \
#                 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
def PlotContourMap(fname, maxlat, minlat, maxlon, minlon, \
    dlon=0.5, dlat=0.5, title='', datatype='ph', outfname='', res='i',\
    browseflag=True, saveflag=False, mapflag='regional_ortho', mapfactor=2.75):
    """
    Plot Tomography Map
    longitude latidute ZValue
    """
    if title=='':
        title=fname;
    if outfname=='':
        outfname=fname;
    Inarray=np.loadtxt(fname)
    LonLst=Inarray[:,0]
    LatLst=Inarray[:,1]
    ZValue=Inarray[:,2]
    llcrnrlon=LonLst.min()
    llcrnrlat=LatLst.min()
    urcrnrlon=LonLst.max()
    urcrnrlat=LatLst.max()
    Nlon=int((urcrnrlon-llcrnrlon)/dlon)+1
    Nlat=int((urcrnrlat-llcrnrlat)/dlat)+1
    lon_min=minlon-1
    lat_min=minlat
    lon_max=maxlon
    lat_max=maxlat
    lat_centre = (lat_max+lat_min)/2.0
    lon_centre = (lon_max+lon_min)/2.0
    fig=plt.figure(num=None, figsize=(8, 12), dpi=100, facecolor='w', edgecolor='k')
    # ax=plt.subplot(111)
    # m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, \
    #     rsphere=(6378137.00,6356752.3142), resolution='c', projection='merc')
    if mapflag=='global':
        m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
        m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
        m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
    elif mapflag=='regional_ortho':
        m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
        m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
            llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/mapfactor)
        # labels = [left,right,top,bottom]
        m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,0])
        m.drawmeridians(np.arange(-170.0,170.0,10.0))	
    elif mapflag=='regional_merc':
        m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
        m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,0,1])
        m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])	
        #m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
        #m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
    
    lon = LonLst[(LatLst<maxlat-0.5)*(LatLst>minlat)]
    lat = LatLst[(LatLst<maxlat-0.5)*(LatLst>minlat)]
    ZValue=ZValue[(LatLst<maxlat-0.5)*(LatLst>minlat)]
    x,y = m(lon, lat)
    xi = np.linspace(x.min(), x.max(), Nlon)
    yi = np.linspace(y.min(), y.max(), Nlat)
    xi, yi = np.meshgrid(xi, yi)
    #-- Interpolating at the points in xi, yi
    zi = griddata(x, y, ZValue, xi, yi)
    #m.pcolormesh(xi, yi, zi, cmap='jet', shading='gouraud')
    m.pcolormesh(xi, yi, zi, cmap='gist_rainbow', shading='gouraud')
    # print lat.max()
    # m.imshow(zi, cmap='gist_rainbow')
    cb=m.colorbar(location='bottom',size='2%')
    cb.ax.tick_params(labelsize=15)
    cb.set_label('Phase travel time (sec)', fontsize=20)
    #cb.set_label('Amplitude (nm)', fontsize=20)
    # m.colorbar(location='bottom',size='2%')
    levels=np.linspace(ZValue.min(), ZValue.max(), 40)
    m.contour(xi, yi, zi, colors='k', levels=levels)
    m.drawcoastlines()
    
    blon=np.arange(100)*(maxlon-minlon)/100.+minlon;
    blat=np.arange(100)*(maxlat-minlat)/100.+minlat;
    Blon=blon;
    Blat=np.ones(Blon.size)*minlat;
    x,y = m(Blon, Blat)
    m.plot(x, y, 'b-', lw=3)
    
    Blon=blon;
    Blat=np.ones(Blon.size)*maxlat;
    x,y = m(Blon, Blat)
    m.plot(x, y, 'b-', lw=3)
    
    Blon=np.ones(Blon.size)*minlon;
    Blat=blat;
    x,y = m(Blon, Blat)
    m.plot(x, y, 'b-', lw=3)
    
    Blon=np.ones(Blon.size)*maxlon;
    Blat=blat;
    x,y = m(Blon, Blat)
    m.plot(x, y, 'b-', lw=3)
    plt.suptitle('Travel time map (10 sec)', fontsize=22);
    #plt.suptitle('Amplitude field (10 sec)', fontsize=22);
    if browseflag==True:
        plt.draw()
        plt.pause(1) # <-------
        raw_input("<Hit Enter To Close>")
        plt.close('all')
    if saveflag==True:
        fig.savefig(outfname+'.pdf', format='pdf')
    return 
    
minlat=25.;
maxlat=52.;
minlon=90.;
maxlon=143.;
PlotContourMap(fname,mapflag='regional_merc', maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat, dlon=dlon, dlat=dlat)

#!/usr/bin/env python

