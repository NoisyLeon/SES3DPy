
"""
Adaptable Tomographic Model Format (atmf): A hdf5 database to store and manipulate 3D Earth model

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import h5py
import numpy as np
import glob, os, shutil
import warnings
from functools import partial
import multiprocessing
import numexpr as npr
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from matplotlib.mlab import griddata
import obspy.geodetics
from lasif import colors
from pyproj import Geod


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

class ATMFDataSet(h5py.File):
    """ An object for the storing and manipulating 3D velocity model
    """
    def readRefmodel(self, infname='ak135.mod', modelname='ak135', header={'depth': 0, 'rho':1, 'vp':2, 'vs':3, 'Qp':4, 'Qs':5}):
        """
        Read reference 1D model, default is ak135 model
        ====================================================================
        Input Parameters:
        infname     - input file name
        modelname   - model name
        header      - header information
        ====================================================================
        """
        inArr = np.loadtxt(infname)
        try: self.create_dataset( name = modelname, shape=inArr.shape, data=inArr)
        except RuntimeError: print 'Reference model:',modelname,' already exists!'
        return
    
    def readCVmodel(self, indir, modelname, grdlst='', header={'depth': 0, 'vs':1, 'vp':2, 'rho':3, 'Qs':4},
                minlat=None, Nlat=None, dlat=None, minlon=None, Nlon=None,  dlon=None, sfx='_mod'):
        """
        Read 3D velocity model assuming Weisen's format
        ===============================================================================================
        Input Parameters:
        indir           - input directory
        modelname       - model name
        grdlst          - grid list( name longitude latitude )
        header          - header information
        dlon, dlat      - grid interval
        Nlon, Nlat      - grid number in longitude, latitude
        minlat, minlon  - minimum latitude/longitude
        sfx             - input model file suffix
        -----------------------------------------------------------------------------------------------
        Three options to read 3D model:
        1. grid list file is defined, will read model according to grdlst
        2. dlon, dlat, Nlon, Nlat, minlat, minlon are defined, will read model accoring to
            longitude/latitude arrays generated from those parameters
        3. Else, will read all the files in the input directory
        ===============================================================================================
        """
        group=self.require_group( name = modelname )
        for hkey in header.keys(): group.attrs.create(name = hkey, data=header[hkey], dtype='i')
        if os.path.isfile(grdlst): self._read_from_lst(indir=indir, grdlst=grdlst, group=group)
        elif minlat!=None and Nlat!=None and minlon!=None and Nlon!=None and dlat!=None and dlon!=None:
            self._read_from_lat_lon(indir=indir, group=group, minlat=minlat, Nlat=Nlat, dlat=dlat, minlon=minlon, Nlon=Nlon, dlon=dlon, sfx=sfx)
        else: self._read_from_dir(indir=indir, group=group, sfx=sfx)
        return
                
            
    def _read_from_lst(self, indir, grdlst , group):
        """
        Read tomographic model according to grid list
        name longitude latitude
        """
        with open(grdlst, 'r') as f:
            Name=[]
            for lines in f.readlines():
                lines=lines.split()
                name=lines[0]
                lon=float(lines[1])
                lat=float(lines[2])
                if Name.__contains__(name):
                    index=Name.index(name)
                    if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                        raise ValueError('Incompatible grid point:' + name+' in grid list!')
                    else:
                        print 'Warning: Repeated grid point:' +name+' in grid list!'
                        continue
                Name.append(name)
                infname=indir+'/'+name+'_mod'
                if not os.path.isfile(infname):
                    warnings.warn('profile: lon='+str(lon)+' lat='+str(lat)+' not exists!', UserWarning, stacklevel=1)
                    continue
                inArr = np.loadtxt(infname)
                dset = group.create_dataset( name=name, shape=inArr.shape, data=inArr)
                dset.attrs.create(name = 'lon', data=lon, dtype='f')
                dset.attrs.create(name = 'lat', data=lat, dtype='f')
        return
    
    def _read_from_lat_lon(self, indir, group, minlat, Nlat, dlat, minlon, Nlon, dlon, sfx='_mod'):
        """
        Read tomographic model according to longitude/latitude array
        """
        latArr=minlat+np.arange(Nlat)*dlat
        lonArr=minlon+np.arange(Nlon)*dlon
        for lat in latArr:
            for lon in lonArr:
                name='%g_%g' %(lon, lat)
                infname=indir+'/'+name+sfx
                if not os.path.isfile(infname):
                    warnings.warn('profile: lon='+str(lon)+' lat='+str(lat)+' not exists!', UserWarning, stacklevel=1)
                    continue
                inArr = np.loadtxt(infname)
                dset = group.create_dataset( name=name, shape=inArr.shape, data=inArr)
                dset.attrs.create(name = 'lon', data=lon, dtype='f')
                dset.attrs.create(name = 'lat', data=lat, dtype='f')
        return
    
    def _read_from_dir(self, indir, group, sfx='_mod'):
        """
        Read tomographic model from all the files in the input directory
        """
        for infname in glob.glob(indir+'/*'+sfx):
            f1=infname.split(sfx)[0]
            f1=f1.split('/')[-1]
            f2=f1.split('_')
            lon = float(f2[0])
            lat = float(f2[1])
            if lon > 180.: lon-=360. # NOTE!!!
            name = '%g_%g' %(lon, lat)
            inArr = np.loadtxt(infname)
            dset = group.create_dataset( name=name, shape=inArr.shape, data=inArr)
            dset.attrs.create(name = 'lon', data=lon, dtype='f')
            dset.attrs.create(name = 'lat', data=lat, dtype='f')
        return
    
    def vertical_extend(self, inname, outname, block=True, refname ='ak135', dz=[0.5, 1], depthlst=[50, 200], z0lst =[],  maxdepth = 410.,
                    outdir = None, dz2=10., dz3= 50., verbose=True, useak135Q=True ):
        """
        Vertical profile extension
        Only suitable when input 3D model is upper mantle model( zmax < 410km)
        ================================================================================================
        Algorithmn:
        step 1: interpolate the profile with given interval( dz ) and depth range (depthlst)
        step 2: extend to 410 km / maxdepth ( if maxdepth < 410 km) in ak135 model
        step 3: extend to maxdepth ( if > 410 km) in ak135 model
        Note: step 2 and 3 need further test !!!
        ------------------------------------------------------------------------------------------------
        Input Parameters:
        inname       - input model(group) name
        outname      - output model(group) name
        block        - output is block model or not
        refname      - reference model(group) name
        dz           - list for depth interval and bottom depth (grid points)
        depthlst     - list for bottom depth (grid points)
        z0lst        - list for top depth point (0 for grid model, non-zero for block model)
        maxdepth     - maximum depth to extend (grid points)
        outdir       - txt output directory( default = None)
        header       - header information
        dz2, dz3     - depth interval for step 2, 3
        ------------------------------------------------------------------------------------------------
        Output:
        self[outname].attrs
                    ['depthArr'] - depth array for output
                    ['dz']       - depth intervals
                    ['depth']    - bottom depth list
                    ['...']      - index information
        self[outname/lon_lat]    - interpolated data
                    attrs: 'lon', 'lat'
        =================================================================================================
        """
        # Generate numpy array for depth interpolation
        header={'depth': 0, 'vs':1, 'vp':2, 'rho':3, 'Qs':4}
        dz = np.asarray(dz)
        depthlst = np.asarray(depthlst)
        if len(dz) == 0 or len(depthlst) == 0 or len(dz) != len(depthlst) :  raise ValueError('Error input for depth!')
        if len(z0lst) != len(dz):
            if block:
                print 'NOTE : Output is block model!'
                z0lst = np.ones(len(depthlst)) * dz / 2.
            else:
                print 'NOTE : Output is grid model!'
                z0lst = np.zeros(len(depthlst)) 
        depthArr = np.array([], dtype='float')
        for i in xrange(len(dz)):
            idz = dz[i]
            iz = depthlst[i]
            if i == 0:
                Nz = int(iz/idz) 
                izArr = np.arange(Nz) * idz + z0lst[i]
            else:
                Nz = int((iz-depthlst[i-1])/idz) 
                izArr = np.arange(Nz) * idz + depthlst[i-1] + z0lst[i]
            if abs(izArr[-1] - iz + idz - z0lst[i]) > 0.001:
                warnings.warn('Not exact division for depth: '+str(iz) + ' km', UserWarning, stacklevel=1)
            depthArr=np.append(depthArr, izArr)
        if not block: depthArr=np.append(depthArr, depthlst[-1])
        # create group name
        group=self.require_group( name = outname )
        dzAttrs=np.array(dz)
        depthAttrs=np.array(depthlst)
        # determine whether to extend to 410km/maxdepth(< 410km) or not
        if (maxdepth > (depthArr[-1] + z0lst[-1]) and (maxdepth - depthArr[-1] - z0lst[-1]) % dz2 ==0.):
            step2 = True
            if maxdepth > 410.:
                dzAttrs=np.append(dzAttrs, dz2)
                depthAttrs=np.append(depthAttrs, 410.)
            else:
                dzAttrs=np.append(dzAttrs, dz2)
                depthAttrs=np.append(depthAttrs, maxdepth)
        else: step2 = False
        # determine whether to extend to maxdepth(>410km) or not
        if maxdepth > 410. and (maxdepth-410.)%dz3 == 0:
            step3 = True
            dzAttrs=np.append(dzAttrs, dz3)
            depthAttrs=np.append(depthAttrs, maxdepth)
        else: step3 = False
        ak135Arr    = self[refname][...]
        depthak135  = ak135Arr[:,0]
        RhoAk135    = ak135Arr[:,1]
        VpAk135     = ak135Arr[:,2]
        VsAk135     = ak135Arr[:,3]
        QAk135      = ak135Arr[:,5]
        # write header information
        for hkey in header.keys(): group.attrs.create(name = hkey, data=header[hkey], dtype='i')
        group.attrs.create(name = 'dz', data=dzAttrs, dtype='f')
        group.attrs.create(name = 'depth', data=depthAttrs, dtype='f')
        if block: group.attrs.create(name = 'isblock', data=1, dtype='i')
        else: group.attrs.create(name = 'isblock', data=0, dtype='i')
        # loop over each profiles
        for index in self[inname].keys():
            print 'Vertical extension for:', index
            # step 1
            depthInter = np.copy(depthArr)
            inArr   = self[inname][index][...]
            depth   = inArr[:,0]
            Vs      = inArr[:,1]
            Vp      = inArr[:,2]
            Rho     = inArr[:,3]
            # Interpolate to depthInter array for each profile
            VpInter = np.interp(depthInter, depth, Vp)
            VsInter = np.interp(depthInter, depth, Vs)
            RhoInter= np.interp(depthInter, depth, Rho)
            if not useak135Q:
                Q       = inArr[:,4]
                QInter  = np.interp(depthInter, depth, Q)
            if step2:
                # Extend to ak135 410km/maxdepth with linear trend, with interval of dz1
                Vsak135_410 = 4.8702
                Vpak135_410 = 9.0302
                Rhoak135_410= 3.5068
                Qak135_410  = 146.57
                if maxdepth > 410.:
                    if block:
                        depInter2=depthlst[-1]+np.arange((410. - depthlst[-1])/dz2)*dz2 + dz2/2. # depth array for interpolation
                    else:
                        depInter2=depthlst[-1]+np.arange((410. - depthlst[-1])/dz2)*dz2 + dz2 # depth array for interpolation
                else:
                    if block:
                        depInter2=depthlst[-1]+np.arange((maxdepth - depthlst[-1])/dz2)*dz2  + dz2/2.# depth array for interpolation
                    else:
                        depInter2=depthlst[-1]+np.arange((maxdepth - depthlst[-1])/dz2)*dz2 +dz2 # depth array for interpolation
                # data for interpolation
                Vsak135_2   = np.append(Vs[-1], Vsak135_410)
                Vpak135_2   = np.append(Vp[-1], Vpak135_410)
                Rhoak135_2  = np.append(Rho[-1], Rhoak135_410)
                if not useak135Q: Qak135_2    = np.append(Q[-1], Qak135_410)
                depthak135_2= np.append(depth[-1], 410.)
                # interpolate for step 2
                VsInter2    = np.interp(depInter2, depthak135_2, Vsak135_2)
                VpInter2    = np.interp(depInter2, depthak135_2, Vpak135_2)
                RhoInter2   = np.interp(depInter2, depthak135_2, Rhoak135_2)
                if not useak135Q: QInter2     = np.interp(depInter2, depthak135_2, Qak135_2)
                # append data
                depthInter  = np.append(depthInter, depInter2)
                VsInter     = np.append(VsInter,VsInter2)
                VpInter     = np.append(VpInter,VpInter2)
                RhoInter    = np.append(RhoInter,RhoInter2)
                if not useak135Q: QInter      = np.append(QInter,QInter2)
            if step3:
                # Interpolate to maxdepth if it is larger than 410 km 
                Vsak135_3   = VsAk135[(depthak135>410.)]
                Vpak135_3   = VpAk135[(depthak135>410.)]
                Rhoak135_3  = RhoAk135[(depthak135>410.)]
                if not useak135Q: Qak135_3    = QAk135[(depthak135>410.)]
                depthak135_3= depthak135[(depthak135>410.)]
                if block:
                    depInter3=410.+np.arange((maxdepth-410.)/dz3)*dz3+dz3/2.
                else:
                    depInter3=410.+np.arange((maxdepth-410.)/dz3)*dz3+dz3
                # interpolate for step 3
                VsInter3    = np.interp(depInter3, depthak135_3, Vsak135_3)
                VpInter3    = np.interp(depInter3, depthak135_3, Vpak135_3)
                RhoInter3   = np.interp(depInter3, depthak135_3, Rhoak135_3)
                if not useak135Q: QInter3     = np.interp(depInter3, depthak135_3, Qak135_3)
                # append data
                depthInter  = np.append(depthInter, depInter3)
                VsInter     = np.append(VsInter,VsInter3)
                VpInter     = np.append(VpInter,VpInter3)
                RhoInter    = np.append(RhoInter,RhoInter3)
                if not useak135Q: QInter      = np.append(QInter,QInter3)
            if useak135Q: QInter = np.interp(depthInter, depthak135, QAk135)
            # save data
            L=depthInter.size
            outArr=np.append(depthInter, VsInter)
            outArr=np.append(outArr, VpInter)
            outArr=np.append(outArr, RhoInter)
            outArr=np.append(outArr, QInter)
            outArr=outArr.reshape((5, L))
            outArr=outArr.T
            if outdir !=None:
                # save to txt file
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                np.savetxt(outdir+'/'+index+'_mod', outArr, fmt='%g')
            lonlat=index.split('_')
            lon = float(lonlat[0])
            lat = float(lonlat[1])
            # save to atmf dataset
            try:
                dset = group.create_dataset( name=index, shape=outArr.shape, data=outArr)
                dset.attrs.create(name = 'lon', data=lon, dtype='f')
                dset.attrs.create(name = 'lat', data=lat, dtype='f')
            except RuntimeError: pass
        group.attrs.create(name = 'depthArr', data=depthInter, dtype='f')
        return
    
    def getavg(self, modelname ):
        """
        Get average velocity profile, given model name
        """
        depthArr = self[modelname].attrs['depthArr']
        Vs  = np.ones(depthArr.size)
        Vp  = np.ones(depthArr.size)
        Rho = np.ones(depthArr.size)
        Q   = np.ones(depthArr.size)
        Nm  = len(self[modelname].keys())
        for index in self[modelname].keys():
            inArr   = self[modelname][index][...]
            Vs      = Vs + inArr[:,1]
            Vp      = Vp + inArr[:,2]
            Rho     = Rho + inArr[:,3]
            Q       = Q + inArr[:,4]
        Vs  = Vs / Nm
        Vp  = Vp / Nm
        Rho = Rho / Nm
        Q   = Q / Nm
        outArr=np.append(depthArr, Vs)
        outArr=np.append(outArr, Vp)
        outArr=np.append(outArr, Rho)
        outArr=np.append(outArr, Q)
        outArr=outArr.reshape((5, depthArr.size))
        outArr=outArr.T
        self[modelname].attrs.create( name='avg_model', shape=outArr.shape, data=outArr)
        return
            
    def minmaxlonlat(self, modelname):
        """Get minimum/maximum latitude and longitude, given model name
        """
        minlon = 999
        maxlon = -999
        minlat = 999
        maxlat = -999
        for index in self[modelname].keys():
            minlon = min( self[modelname][index].attrs['lon'], minlon)
            maxlon = max( self[modelname][index].attrs['lon'], maxlon)
            minlat = min( self[modelname][index].attrs['lat'], minlat)
            maxlat = max( self[modelname][index].attrs['lat'], maxlat)
        print 'latitude:',minlat,maxlat,'longitude:',minlon,maxlon
        return
    
    def horizontal_extend_old(self, modelname, minlat, maxlat, dlat, minlon, maxlon, dlon, sfx='_ses3d', Dref=500., outdir = None):
        """
        Horizontal extension, old version, relatively slow
        ================================================================================================
        Algorithmn:
        Find the closest geographical point and extend the vertical profile using a linear interpolation
        between the closest profile and average profile.
        ------------------------------------------------------------------------------------------------
        Input Parameters:
        modelname       - input model(group) name
        indir           - input directory
        modelname       - model name
        dlon, dlat      - grid interval
        minlat, minlon  - minimum latitude/longitude
        maxlat, maxlon  - maximum latitude/longitude
        sfx             - output model name suffix
        Dref            - reference distance for interpolation
        outdir          - output directory for txt model files
        =================================================================================================
        """
        from geopy.distance import great_circle
        avgArr = self[modelname].attrs['avg_model'][...]
        latarr = minlat + np.arange( (maxlat-minlat)/dlat + 1)*dlat
        lonarr = minlon + np.arange( (maxlon-minlon)/dlon + 1)*dlon
        outname = modelname+sfx
        try: del self[outname]
        except: pass
        self.copy( source = modelname, dest = outname )
        latlon_lst= []
        for index in self[outname].keys():
            elat = self[outname][index].attrs['lat']
            elon = self[outname][index].attrs['lon']
            name = '%g'%(elon) + '_%g' %(elat)
            latlon_lst.append([elat, elon])
        print '================ Start horizontal extrapolation =============='
        for lon in lonarr:
            for lat in latarr:
                mindist=99999
                clon=0
                clat=0
                name='%g'%(lon) + '_%g' %(lat)
                if (name in self[outname].keys()): continue
                print 'Extending to ', name
                for latlon in latlon_lst:
                    elat = latlon[0]
                    elon = latlon[1]
                    dist=great_circle((elat, elon),(lat, lon)).km
                    if dist<mindist:
                        mindist=dist
                        clon=elon
                        clat=elat
                    if (abs(elat - lat) <=dlat and abs (elon-lon) == dlon) or (abs(elat - lat) ==dlat and abs (elon-lon) <= dlon): break
                cname = '%g'%(clon) + '_%g' %(clat)
                cArr = self[outname][cname][...]
                if mindist < Dref: outArr=((Dref-mindist)*cArr+mindist*avgArr)/Dref
                    # outArr=npr.evaluate ( '((Dref-mindist)*cArr+mindist*avgArr)/Dref ')
                else: outArr=avgArr
                dset = self[outname].create_dataset( name=name, shape=outArr.shape, data=outArr)
                dset.attrs.create(name = 'lon', data=lon, dtype='f')
                dset.attrs.create(name = 'lat', data=lat, dtype='f')
                # save to txt file
                if outdir !=None:
                    if not os.path.isdir(outdir): os.makedirs(outdir)
                    np.savetxt(outdir+'/'+name+'_mod', outArr, fmt='%g')
        self[outname].attrs.create(name = 'minlat', data=minlat, dtype='f')
        self[outname].attrs.create(name = 'maxlat', data=maxlat, dtype='f')
        self[outname].attrs.create(name = 'dlat', data=dlat, dtype='f')
        self[outname].attrs.create(name = 'minlon', data=minlon, dtype='f')
        self[outname].attrs.create(name = 'maxlon', data=maxlon, dtype='f')
        self[outname].attrs.create(name = 'dlon', data=dlon, dtype='f')
        print '================ End horizontal extrapolation =============='
        return
    
    def horizontal_extend(self, modelname, minlat, maxlat, dlat, minlon, maxlon, dlon, sfx='_ses3d', Dref=500., outdir = None):
        """
        Horizontal extension
        ================================================================================================
        Algorithmn:
        Find the closest geographical point and extend the vertical profile using a linear interpolation
        between the closest profile and average profile.
        ------------------------------------------------------------------------------------------------
        Input Parameters:
        modelname       - input model(group) name
        indir           - input directory
        modelname       - model name
        dlon, dlat      - grid interval
        minlat, minlon  - minimum latitude/longitude
        maxlat, maxlon  - maximum latitude/longitude
        sfx             - output model name suffix
        Dref            - reference distance for interpolation
        outdir          - output directory for txt model files
        =================================================================================================
        """
        avgArr = self[modelname].attrs['avg_model'][...]
        latarr = minlat + np.arange( (maxlat-minlat)/dlat + 1)*dlat
        lonarr = minlon + np.arange( (maxlon-minlon)/dlon + 1)*dlon
        outname = modelname+sfx
        try: del self[outname]
        except: pass
        self.copy( source = modelname, dest = outname )
        elonArr = np.array([]); elatArr = np.array([])
        for index in self[outname].keys():
            elat = self[outname][index].attrs['lat']
            elon = self[outname][index].attrs['lon']
            elonArr = np.append(elonArr, elon)
            elatArr = np.append(elatArr, elat)
        L=elonArr.size
        g = Geod(ellps='WGS84')
        print '================ Start horizontal extrapolation =============='
        for lon in lonarr:
            for lat in latarr:
                name='%g'%(lon) + '_%g' %(lat)
                if (name in self[outname].keys()): continue
                print 'Extending to ', name
                clonArr=np.ones(L)*lon; clatArr=np.ones(L)*lat
                az, baz, dist = g.inv(clonArr, clatArr, elonArr, elatArr)
                imin = dist.argmin(); mindist = dist[imin]
                elat = elatArr[imin]; elon = elonArr[imin]
                cname = '%g'%(elon) + '_%g' %(elat)
                cArr = self[outname][cname][...]
                if mindist < Dref: outArr=((Dref-mindist)*cArr+mindist*avgArr)/Dref
                    # outArr=npr.evaluate ( '((Dref-mindist)*cArr+mindist*avgArr)/Dref ')
                else: outArr=avgArr
                dset = self[outname].create_dataset( name=name, shape=outArr.shape, data=outArr)
                dset.attrs.create(name = 'lon', data=lon, dtype='f')
                dset.attrs.create(name = 'lat', data=lat, dtype='f')
                # save to txt file
                if outdir !=None:
                    if not os.path.isdir(outdir): os.makedirs(outdir)
                    np.savetxt(outdir+'/'+name+'_mod', outArr, fmt='%g')
        self[outname].attrs.create(name = 'minlat', data=minlat, dtype='f')
        self[outname].attrs.create(name = 'maxlat', data=maxlat, dtype='f')
        self[outname].attrs.create(name = 'dlat', data=dlat, dtype='f')
        self[outname].attrs.create(name = 'minlon', data=minlon, dtype='f')
        self[outname].attrs.create(name = 'maxlon', data=maxlon, dtype='f')
        self[outname].attrs.create(name = 'dlon', data=dlon, dtype='f')
        print '================ End horizontal extrapolation =============='
        return
    
    
    
    def getmoho(self, groupname, vmin=10, vmax=70, sigma=4,
            minlat=-999, maxlat=999, minlon=-999, maxlon=999, mindepth=10.,projection='lambert' ):
        """
        Read hdf5 model
        ================================================================================================
        Input parameters:
        infname         - input filename
        groupname       - group name
        minlon, maxlon  - defines study region, default is to read corresponding data from hdf5 file 
        minlat, maxlat  -
        maxdepth        - maximum depth to be truncated
        ================================================================================================
        """
        # get latitude/longitude information
        if minlat < self[groupname].attrs['minlat']:
            minlat = self[groupname].attrs['minlat']
        if maxlat > self[groupname].attrs['maxlat']:
            maxlat = self[groupname].attrs['maxlat']
        if minlon < self[groupname].attrs['minlon']:
            minlon = self[groupname].attrs['minlon']
        if maxlon > self[groupname].attrs['maxlon']:
            maxlon = self[groupname].attrs['maxlon']
        dlon = self[groupname].attrs['dlon']
        dlat = self[groupname].attrs['dlat']
        lonArr=minlon+np.arange((maxlon-minlon)/dlon+1)*dlon
        latArr=minlat+np.arange((maxlat-minlat)/dlat+1)*dlat
        lons, lats=np.meshgrid(lonArr, latArr)
        mohoArr = np.ones(lons.shape)
        for ilon in xrange(lonArr.size):
            for ilat in xrange(latArr.size):
                lon=lonArr[ilon]
                lat=latArr[ilat]
                name='%g_%g' %(lon, lat)
                Vprofile = self[groupname][name].value
                depthArr = Vprofile[:,0]
                QArr = Vprofile[:,4]
                zinterp = np.arange(200)*0.5 + mindepth
                Qinterp = np.interp(zinterp, depthArr, QArr)
                delQ = abs(Qinterp[1:] - Qinterp[:-1])
                mohoArr[ilat, ilon] = zinterp[delQ.argmax()]
        lon_min=minlon
        lat_min=minlat
        lon_max=maxlon
        lat_max=maxlat
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
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
             distEW, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                                 lat_min, lon_max) # distance is in m
             distNS, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                                lat_max+1.7, lon_min) # distance is in m
             m = Basemap(width=distEW, height=distNS,
             rsphere=(6378137.00,6356752.3142),\
             resolution='l', projection='lcc',\
             lat_1=lat_min, lat_2=lat_max, lat_0=lat_centre+1.2, lon_0=lon_centre)
             m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=2, dashes=[2,2], labels=[1,0,0,0], fontsize=15)
             m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=2, dashes=[2,2], labels=[0,0,1,1], fontsize=15)
        moho_filtered=mohoArr.copy()
        for iteration in xrange(int(sigma)):
            for i in np.arange(1,latArr.size-1):
                for j in np.arange(1,lonArr.size-1):
                    moho_filtered[i,j]=(mohoArr[i,j]+mohoArr[i+1,j]+mohoArr[i-1,j]+mohoArr[i,j+1]+mohoArr[i,j-1])/5.0
        x,y = m(lons, lats)
        m.drawcoastlines()
        cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
        cmap =discrete_cmap(int(vmax-vmin)/5, cmap)
        im=m.pcolormesh(x, y, moho_filtered, shading='gouraud', cmap=cmap, vmin=10, vmax=70)
        cb = m.colorbar(im,"right", size="3%", pad='2%', ticks=np.arange( (vmax-vmin)/5+1)*5+vmin)
        cb.set_label('km', fontsize=20, rotation=90)
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        plt.show()
        return moho_filtered
    
    def vprofile2txt(self, modelname, lon, lat, outfname=None):
        name='%g'%(lon) + '_%g' %(lat)
        outArr = self[modelname][name][...]
        if outfname == None: outfname = name+'_mod'
        np.savetxt(outfname, outArr, fmt='%g')
        return
    
    
