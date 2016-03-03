import numpy as np
import matplotlib.pyplot as plt
import os,sys;
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
import matplotlib.cm
from matplotlib.ticker import FuncFormatter
# sys.path.append('/projects/life9360/.local/quickkey')
import colormaps as cm
# foo.MyClass()

fname=sys.argv[1]
dlon=float(sys.argv[2])
dlat=float(sys.argv[3])
# my_colormap=cm.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0],\
#                 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], \
#                 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
def PlotTomoMap(fname, dlon=0.5, dlat=0.5, title='', datatype='ph', outfname='', browseflag=True, saveflag=False):
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
    fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, \
        rsphere=(6378137.00,6356752.3142), resolution='l', projection='merc')
    
    lon = LonLst
    lat = LatLst
    x,y = m(lon, lat)
    xi = np.linspace(x.min(), x.max(), Nlon)
    yi = np.linspace(y.min(), y.max(), Nlat)
    xi, yi = np.meshgrid(xi, yi)
    #-- Interpolating at the points in xi, yi
    zi = griddata(x, y, ZValue, xi, yi)
    # m.pcolormesh(xi, yi, zi, cmap='seismic_r', shading='gouraud')
    cmap=matplotlib.cm.seismic_r
    cmap.set_bad('w',1.)
    m.imshow(zi, cmap='rainbow')
    m.drawcoastlines()
    m.colorbar(location='bottom',size='2%')
    # m.fillcontinents()
    # draw parallels
    m.drawparallels(np.arange(-90,90,10),labels=[1,1,0,1])
    # draw meridians
    m.drawmeridians(np.arange(-180,180,10),labels=[1,1,1,0])
    plt.suptitle(title,y=0.9, fontsize=22);
    if browseflag==True:
        plt.draw()
        plt.pause(1) # <-------
        raw_input("<Hit Enter To Close>")
        plt.close('all')
    if saveflag==True:
        fig.savefig(outfname+'.ps', format='ps')
    return 
    

PlotTomoMap(fname,dlon=dlon, dlat=dlat)#!/usr/bin/env python

