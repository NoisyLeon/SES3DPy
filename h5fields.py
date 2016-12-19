# -*- coding: utf-8 -*-
"""
A python module for SES3D hdf5 field file manipulation.
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import os
import re
import glob
import numpy as np
from lasif import rotations
from lasif import colors
import matplotlib.pylab as plt 
from mpl_toolkits.basemap import Basemap
from functools import partial
import multiprocessing
from pylab import savefig
from geopy.distance import great_circle
import obspy
import h5py
from sympy.ntheory import primefactors
from pyproj import Geod
import warnings

#- Pretty units for some components.
UNIT_DICT = {
    "vp": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "vsv": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "vsh": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "rho": r"$\frac{\mathrm{kg}^3}{\mathrm{m}^3}$",
    "rhoinv": r"$\frac{\mathrm{m}^3}{\mathrm{kg}^3}$",
    "vx": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "vy": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
    "vz": r"$\frac{\mathrm{m}}{\mathrm{s}}$",
}



class proc_data(object):
    def __init__(self, field=np.array([]), lon=np.array([]), lat=np.array([])):
        self.field=field
        self.lon=lon
        self.lat=lat

class snap_data(object):
    """
    """
    def __init__(self, proc_datas=None):
        self.proc_datas=[]
        if isinstance(proc_datas, proc_data):
            proc_datas = [proc_datas]
        if proc_datas:
            self.proc_datas.extend(proc_datas)

    def __add__(self, other):
        """
        Add two snap_data with self += other.
        """
        if isinstance(other, proc_data):
            other = snap_data([other])
        if not isinstance(other, snap_data):
            raise TypeError
        proc_datas = self.proc_datas + other.proc_datas
        return self.__class__(proc_datas=proc_datas)

    def __len__(self):
        """
        Return the number of proc_datas in the snap_data object.
        """
        return len(self.proc_datas)

    def __getitem__(self, index):
        """
        __getitem__ method of snap_data objects.
        :return: proc_data objects
        """
        if isinstance(index, slice):
            return self.__class__(proc_datas=self.proc_datas.__getitem__(index))
        else:
            return self.proc_datas.__getitem__(index)

    def append(self, procdata):
        """
        Append a single proc_data object to the current snap_data object.
        """
        if isinstance(procdata, proc_data):
            self.proc_datas.append(procdata)
        else:
            msg = 'Append only supports a single proc_data object as an argument.'
            raise TypeError(msg)
        return self

class h5fields(h5py.File):
    
    def convert_to_vts(self, outdir, component, iter0=None, iterf=None, diter=None, Radius=0.98, verbose=True):
        """
        Plot depth slices of field component at given depth ranging between "valmin" and "valmax"
        ================================================================================================
        Input parameters:
        outdir          - output directory
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        depth           - depth for plot (km)
        iter0, iterf    - start/end iteration index
        diter           - iteration interval
        Radius          - radius for output sphere
        =================================================================================================
        """
        if not os.path.isdir(outdir): os.makedirs(outdir)
        group=self[component]
        from tvtk.api import tvtk, write_data
        try: iterArr=np.arange(iter0 ,iterf+diter, diter, dtype=int)
        except: iterArr = group.keys()
        least_prime=None
        for iteration in iterArr:
            subgroup=group[str(iteration)]
            if len(subgroup.keys())==0: continue
            if verbose: print 'Output vts file for iteration =',iteration
            theta=np.array([]); phi=np.array([]); r=np.array([]); field = np.array([])
            for key in subgroup.keys():
                subdset = subgroup[key]
                field   = np.append(field, (subdset[...]))
                theta1  = subdset.attrs['theta']
                phi1    = subdset.attrs['phi']
                theta1, phi1 = np.meshgrid(theta1, phi1, indexing='ij')
                theta   = np.append(theta, theta1)
                phi     = np.append(phi, phi1)
            x = Radius * np.sin(theta) * np.cos(phi)
            y = Radius * np.sin(theta) * np.sin(phi)
            z = Radius * np.cos(theta)
            if least_prime==None: least_prime=primefactors(field.size)[0]
            dims = (field.size/least_prime, least_prime, 1)
            pts = np.empty(z.shape + (3,), dtype=float)
            pts[..., 0] = x; pts[..., 1] = y; pts[..., 2] = z
            sgrid = tvtk.StructuredGrid(dimensions=dims, points=pts)
            sgrid.point_data.scalars = (field).ravel(order='F')
            sgrid.point_data.scalars.name = component
            outfname=outdir+'/'+component+'_'+str(iteration)+'.vts'
            write_data(sgrid, outfname)
        return
    
    def zero_padding(self, outfname, component, evlo, evla, dt, minV=1.5, iter0=None, iterf=None, diter=None, verbose=True):
        # - Some initialisations. ------------------------------------------------------------------
        g = Geod(ellps='WGS84')
        dset    = h5py.File(outfname)
        dset.attrs.create(name = 'theta_max', data=self.attrs["theta_max"], dtype='f')
        dset.attrs.create(name = 'theta_min', data=self.attrs["theta_min"], dtype='f')
        dset.attrs.create(name = 'phi_min', data=self.attrs["phi_min"], dtype='f')
        dset.attrs.create(name = 'phi_max', data=self.attrs["phi_max"], dtype='f')
        lat_min = 90.0 - self.attrs["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.attrs["theta_min"]*180.0/np.pi
        lon_min = self.attrs["phi_min"]*180.0/np.pi
        lon_max = self.attrs["phi_max"]*180.0/np.pi
        dset.attrs.create(name = 'lat_min', data=lat_min, dtype='f')
        dset.attrs.create(name = 'lat_max', data=lat_max, dtype='f')
        dset.attrs.create(name = 'lon_min', data=lon_min, dtype='f')
        dset.attrs.create(name = 'lon_max', data=lon_max, dtype='f')
        dset.attrs.create(name = 'depth', data=self.attrs['depth'], dtype='f')
        dset.attrs.create(name = 'n_procs', data=self.attrs['n_procs'], dtype='f')
        dset.attrs.create(name = 'rotation_axis', data=self.attrs['rotation_axis'], dtype='f')
        dset.attrs.create(name = 'rotation_angle', data=self.attrs['rotation_angle'], dtype='f')
        group   = dset.create_group( name = component )
        in_group= self[component]
        # - Loop over processor boxes and check if depth falls within the volume. ------------------
        try: iterArr=np.arange(iter0 ,iterf+diter, diter, dtype=int)
        except: iterArr = in_group.keys()
        for iteration in iterArr:
            try: in_subgroup = in_group[str(iteration)]
            except KeyError: continue
            subgroup=group.create_group(name=str(iteration))
            if verbose: print 'Zero padding snapshot for iteration =',iteration
            time = float(iteration) * dt; mindist = time * minV
            for iproc in in_subgroup.keys():
                in_subdset  = in_subgroup[iproc]
                field       = in_subdset.value
                theta       = in_subdset.attrs['theta']
                phi         = in_subdset.attrs['phi']
                lat = 90.-theta/np.pi*180.; lon = phi/np.pi*180.
                lats, lons = np.meshgrid(lat, lon, indexing = 'ij')
                evlaArr = evla * np.ones(lats.shape); evloArr = evlo * np.ones(lats.shape)
                az, baz, distevent = g.inv(lons, lats, evloArr, evlaArr)
                distevent=distevent/1000.
                index_padding  = distevent < mindist
                field[index_padding] = 0
                subdset = subgroup.create_dataset(name=iproc, shape=field.shape, data=field)
                subdset.attrs.create(name = 'theta', data=theta, dtype='f')
                subdset.attrs.create(name = 'phi', data=phi, dtype='f')
        dset.close()
        return
    
    def _get_basemap(self, projection='lambert', resolution='i'):
        """Plot data with contour
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        lat_centre = self.lat_centre; lon_centre = self.lon_centre
        if projection=='merc':
            m=Basemap(projection='merc', llcrnrlat=self.minlat-5., urcrnrlat=self.maxlat+5., llcrnrlon=self.minlon-5.,
                      urcrnrlon=self.maxlon+5., lat_ts=20, resolution=resolution)
            # m.drawparallels(np.arange(self.minlat,self.maxlat,self.dlat), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(self.minlon,self.maxlon,self.dlon), labels=[1,0,0,1])
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m=Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        
        elif projection=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=self.minlon, lat_0=self.minlat, resolution='l')
            m = Basemap(projection='ortho', lon_0=self.minlon, lat_0=self.minlat, resolution=resolution,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz=obspy.geodetics.gps2dist_azimuth(self.minlat, self.minlon,
                                self.minlat, self.maxlon) # distance is in m
            distNS, az, baz=obspy.geodetics.gps2dist_azimuth(self.minlat, self.minlon,
                                self.maxlat+2., self.minlon) # distance is in m
            m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                lat_1=self.minlat, lat_2=self.maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
            # m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=0.5, dashes=[2,2], labels=[1,0,0,0], fontsize=5)
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=0.5, dashes=[2,2], labels=[0,0,0,1], fontsize=5)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries(linewidth=1.)
        
        # m.drawmapboundary(fill_color=[1.0,1.0,1.0])
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        # m.drawlsmask(land_color='0.8', ocean_color='#99ffff')
        m.drawmapboundary(fill_color="white")

        return m
                
    def plot_depth_slice(self, component, vmin, vmax, iteration=0,
            res="l", projection='lambert', zoomin=2, geopolygons=None, evlo=None, evla=None):
        """
        Plot depth slices of field component at given depth ranging between "valmin" and "valmax"
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        vmin, vmax      - minimum/maximum value for plotting
        iteration       - iteration step for snapshot
        res             - resolution of the coastline (c, l, i, h, f)
        proj            - projection type (global, regional_ortho, regional_merc)
        zoomin          - zoom in factor for proj = regional_ortho
        geopolygons     - geological polygons( basins etc. ) for plot
        =================================================================================================
        """
        # - Some initialisations. ------------------------------------------------------------------
        fig=plt.figure()
        self.minlat = self.attrs['lat_min']; self.maxlat = self.attrs['lat_max']
        self.minlon = self.attrs['lon_min']; self.maxlon = self.attrs['lon_max']
        self.n = self.attrs['rotation_axis']; self.rotangle = self.attrs['rotation_angle']
        lat_centre = (self.maxlat+self.minlat)/2.0; lon_centre = (self.maxlon+self.minlon)/2.0
        self.lat_centre, self.lon_centre = rotations.rotate_lat_lon(lat_centre, lon_centre, self.n, -self.rotangle)
        # - Set up the map. ------------------------------------------------------------------------
        m=self._get_basemap(projection=projection)
        try: geopolygons.PlotPolygon(inbasemap=m)
        except: pass
        try:
            evx, evy=m(evlo, evla)
            m.plot(evx, evy, 'yo', markersize=2)
        except: pass
        group=self[component]
        subgroup=group[str(iteration)]
        for key in subgroup.keys():
            subdset = subgroup[key]
            field   = subdset[...]
            theta   = subdset.attrs['theta']
            phi     = subdset.attrs['phi']
            lats    = 90.0 - theta * 180.0 / np.pi
            lons    = phi * 180.0 / np.pi
            lon, lat = np.meshgrid(lons, lats)
            if self.rotangle != 0.0:
                lat_rot = np.zeros(np.shape(lon),dtype=float)
                lon_rot = np.zeros(np.shape(lat),dtype=float)
                for idlon in np.arange(len(lons)):
                    for idlat in np.arange(len(lats)):
                        lat_rot[idlat,idlon],lon_rot[idlat,idlon]  = rotations.rotate_lat_lon(lat[idlat,idlon], lon[idlat,idlon],  self.n, -self.rotangle)
                        lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                lon = lon_rot
                lat = lat_rot
            # - colourmap. ---------------------------------------------------------
            cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
            x, y = m(lon, lat)
            im = m.pcolormesh(x, y, field, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax) 
        # - Add colobar and title. ------------------------------------------------------------------
        cb = m.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        # plt.suptitle("Depth slice of %s at %i km" % (component, r_effective), fontsize=20)
        # - Plot stations if available. ------------------------------------------------------------
        # if self.stations and stations:
        #     x,y = m(self.stlons,self.stlats)
        #     for n in range(self.n_stations):
        #         plt.text(x[n],y[n],self.stnames[n][:4])
        #         plt.plot(x[n],y[n],'ro')
        plt.show()
        print "minimum value: "+str(vmin)+", maximum value: "+str(vmax)
        return
    
    def plot_snapshots(self, component, vmin, vmax, outdir, fprx='wavefield',iter0=200, iterf=17000, \
            diter=200, stations=False, res="i", projection='lambert', dpi=300, zoomin=2, geopolygons=None, evlo=None, evla=None ):
        """
        Plot snapshots of field component at given depth ranging between "valmin" and "valmax"
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        vmin, vmax      - minimum/maximum value for plotting
        outdir          - output directory
        fprx            - output file name prefix
        iter0, iterf    - inital/final iterations for plotting
        diter           - iteration interval
        stations        - plot stations or not
        res             - resolution of the coastline (c, l, i, h, f)
        projection      - projection type (global, regional_ortho, regional_merc)
        dpi             - dots per inch (figure resolution parameter)
        zoomin          - zoom in factor for proj = regional_ortho
        geopolygons     - geological polygons( basins etc. ) for plot
        evlo, evla      - event location for plotting
        =================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        iterArr=np.arange(iter0 ,iterf+diter, diter, dtype=int)
        use_default_iter=False
        for iteration in iterArr:
            if not str(iteration) in self[component].keys():
                warnings.warn('Velocity Snapshot:'+str(iteration)+' does not exist!', UserWarning, stacklevel=1)
                # raise KeyError('Velocity Snapshot:'+str(iteration)+' does not exist!')
                use_in_iter=True
        if use_default_iter: iterArr = self[component].keys()
        self.minlat = self.attrs['lat_min']; self.maxlat = self.attrs['lat_max']
        self.minlon = self.attrs['lon_min']; self.maxlon = self.attrs['lon_max']
        self.n = self.attrs['rotation_axis']; self.rotangle = self.attrs['rotation_angle']
        lat_centre = (self.maxlat+self.minlat)/2.0; lon_centre = (self.maxlon+self.minlon)/2.0
        self.lat_centre, self.lon_centre = rotations.rotate_lat_lon(lat_centre, lon_centre, self.n, -self.rotangle)
        fig=plt.figure(num=None, figsize=(10, 10), dpi=dpi, facecolor='w', edgecolor='k')
        # - Set up the map. ------------------------------------------------------------------------
        m=self._get_basemap(projection=projection)
        try: geopolygons.PlotPolygon(inbasemap=m)
        except: pass
        try:
            evx, evy=m(evlo, evla)
            m.plot(evx, evy, 'yo', markersize=2)
        except: pass
        for iteration in iterArr:
            self._plot_snapshot(inmap=m, component=component, vmin=vmin, vmax=vmax, iteration=iteration, stations=stations)
            outfname=outdir+'/'+fprx+'_%06d.png' %(int(iteration))
            print outfname, outdir
            fig.savefig(outfname, format='png', dpi=dpi)
        return 
    
    def _plot_snapshot(self, inmap, component, vmin, vmax, iteration, stations):
        """Plot snapshot, private function used by make_animation
        """
        print 'Plotting Snapshot for:',iteration,' steps!'
        subgroup=self[component+'/'+str(iteration)]
        for key in subgroup.keys():
            subdset = subgroup[key]
            field   = subdset[...]
            theta   = subdset.attrs['theta']
            phi     = subdset.attrs['phi']
            lats    = 90.0 - theta * 180.0 / np.pi
            lons    = phi * 180.0 / np.pi
            lon, lat = np.meshgrid(lons, lats)
            if self.rotangle != 0.0:
                lat_rot = np.zeros(np.shape(lon),dtype=float)
                lon_rot = np.zeros(np.shape(lat),dtype=float)
                for idlon in np.arange(len(lons)):
                    for idlat in np.arange(len(lats)):
                        lat_rot[idlat,idlon], lon_rot[idlat,idlon]  = rotations.rotate_lat_lon(lat[idlat,idlon], lon[idlat,idlon],  self.n, -self.rotangle)
                        lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                lon = lon_rot
                lat = lat_rot
            # - colourmap. ---------------------------------------------------------
            cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
            x, y = inmap(lon, lat)
            im = inmap.pcolormesh(x, y, field, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax) 
        # - Add colobar and title. ------------------------------------------------------------------
        cb = inmap.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        # - Plot stations if available. ------------------------------------------------------------
        # if (self.stations == True) & (stations==True):
        #     x,y = mymap(self.stlons,self.stlats)
        #     for n in range(self.n_stations):
        #         plt.text(x[n],y[n],self.stnames[n][:4])
        #         plt.plot(x[n],y[n],'ro')
        return
    
    def plot_snapshots_mp(self, component, vmin, vmax, outdir, fprx='wavefield',iter0=100, iterf=17100, diter=200,
            stations=False, res="i", projection='lambert', dpi=300, zoomin=2, geopolygons=None, evlo=None, evla=None, dt=None ):
        """Multiprocessing version of plot_snapshots
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        depth           - depth for plot (km)
        vmin, vmax      - minimum/maximum value for plotting
        outdir          - output directory
        fprx            - output file name prefix
        iter0, iterf    - inital/final iterations for plotting
        diter           - iteration interval
        stations        - plot stations or not
        res             - resolution of the coastline (c, l, i, h, f)
        projection      - projection type (global, regional_ortho, regional_merc)
        dpi             - dots per inch (figure resolution parameter)
        zoomin          - zoom in factor for proj = regional_ortho
        geopolygons     - geological polygons( basins etc. ) for plot
        evlo, evla      - event location for plotting 
        =================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        iterArr=np.arange(iter0 ,iterf+diter, diter, dtype=int)
        use_default_iter=False
        for iteration in iterArr:
            if not str(iteration) in self[component].keys():
                warnings.warn('Velocity Snapshot:'+str(iteration)+' does not exist!', UserWarning, stacklevel=1)
                use_default_iter=True
        if use_default_iter: iterArr = self[component].keys()
        lat_min = self.attrs['lat_min']; lat_max = self.attrs['lat_max']
        lon_min = self.attrs['lon_min']; lon_max = self.attrs['lon_max']
        self.n = self.attrs['rotation_axis']; self.rotangle = self.attrs['rotation_angle']
        lat_centre = (lat_max+lat_min)/2.0; lon_centre = (lon_max+lon_min)/2.0
        lat_centre, lon_centre = rotations.rotate_lat_lon(lat_centre, lon_centre, self.n, -self.rotangle)
        print '================================= Start preparing generating snapshots =================================='
        dataLst=[]
        for iteration in iterArr:
            subgroup=self[component+'/'+str(iteration)]
            snapD=snap_data()
            for key in subgroup.keys():
                subdset = subgroup[key]
                field   = subdset[...]
                theta   = subdset.attrs['theta']
                phi     = subdset.attrs['phi']
                lats    = 90.0 - theta * 180.0 / np.pi
                lons    = phi * 180.0 / np.pi
                snapD.append(proc_data(field=field, lon=lons, lat=lats))
            snapD.n=self.n; snapD.rotangle=self.rotangle; snapD.iteration=iteration
            dataLst.append(snapD)
        self.close()
        print '============================= Start multiprocessing generating snapshots ==============================='
        PLOTSNAP = partial(Iter2snapshot, evlo=evlo, evla=evla, component=component, \
            vmin=vmin, vmax=vmax, stations=stations, fprx=fprx, projection=projection, \
            outdir=outdir, dpi=dpi, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,\
            lat_centre=lat_centre, lon_centre=lon_centre, res=res, \
            zoomin=zoomin, geopolygons=geopolygons, dt=dt)
        pool=multiprocessing.Pool()
        pool.map(PLOTSNAP, dataLst) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print '============================== End multiprocessing generating snapshots ================================='
        return


def Iter2snapshot(snapD, evlo, evla, component, vmin, vmax, stations, fprx, projection, outdir, dpi, \
        lat_min, lat_max, lon_min, lon_max, lat_centre, lon_centre, res, zoomin, geopolygons, dt):
    """Plot snapshot, used by plot_snapshots_mp
    """
    print 'Plotting Snapshot for:',snapD.iteration,' step!'
    # - Set up the map. ------------------------------------------------------------------------
    if projection=='global':
        m=Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
        m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
        m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
    elif projection=='regional_ortho':
        m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
        m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
            llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/zoomin, urcrnry=m1.urcrnry/3.5)
    elif projection=='regional_merc':
        m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
        m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
        m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
    elif projection=='lambert':
        distEW, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                            lat_min, lon_max) # distance is in m
        distNS, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                            lat_max+2., lon_min) # distance is in m
        m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
            lat_1=lat_min, lat_2=lat_max, lon_0=lon_centre, lat_0=lat_centre+1.)
        m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,0,0,0], fontsize=15)
        m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,1], fontsize=15)
    m.drawcoastlines()
    m.fillcontinents(lake_color='#99ffff',zorder=0.2)
    m.drawmapboundary(fill_color="white")
    m.drawcountries()
    try:
        evx, evy=m(evlo, evla)
        m.plot(evx, evy, 'yo', markersize=2)
    except: pass
    for procD in snapD:
        field   = procD.field
        lats    = procD.lat
        lons    = procD.lon
        lon, lat = np.meshgrid(lons, lats)
        if snapD.rotangle != 0.0:
            lat_rot = np.zeros(np.shape(lon),dtype=float)
            lon_rot = np.zeros(np.shape(lat),dtype=float)
            for idlon in np.arange(len(lons)):
                for idlat in np.arange(len(lats)):
                    lat_rot[idlat,idlon], lon_rot[idlat,idlon]  = rotations.rotate_lat_lon(lat[idlat,idlon], lon[idlat,idlon],  snapD.n, -snapD.rotangle)
                    lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
            lon = lon_rot
            lat = lat_rot
        # - colourmap. ---------------------------------------------------------
        cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
        x, y = m(lon, lat)
        im = m.pcolormesh(x, y, field, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax) 
    # - Add colobar and title. ------------------------------------------------------------------
    cb = m.colorbar(im, "right", size="3%", pad='2%')
    if component in UNIT_DICT:
        cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
    try:
        geopolygons.PlotPolygon(inbasemap=m)
    except:
        pass
    outfname=outdir+'/'+fprx+'_%06d.png' %(int(snapD.iteration))
    savefig(outfname, format='png', dpi=dpi)
    del m, lon, lat
    return
    

    
