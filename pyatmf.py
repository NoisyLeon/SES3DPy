
"""
Adaptable Tomographic Model Format (atmf)
"""
import h5py
import numpy as np
import glob, os, shutil
import warnings
from geopy.distance import great_circle
from functools import partial
import multiprocessing
import numexpr as npr


class ATMFDataSet(h5py.File):
    
    def readRefmodel(self, infname, modelname='ak135', header={'depth': 0, 'rho':1, 'vp':2, 'vs':3, 'Qp':4, 'Qs':5}):
        inArr = np.loadtxt(infname)
        try:
            self.create_dataset( name = modelname, shape=inArr.shape, data=inArr)
        except RuntimeError:
            print 'Reference model:',modelname,' already exists!'
        return
    
    def readCVmodel(self, indir, modelname, grdlst='', header={'depth': 0, 'vs':1, 'vp':2, 'rho':3, 'Qs':4},
                minlat=None, Nlat=None, dlat=None, minlon=None, Nlon=None,  dlon=None):
        group=self.require_group( name = modelname )
        for hkey in header.keys():
            group.attrs.create(name = hkey, data=header[hkey], dtype='i')
        if os.path.isfile(grdlst):
            self._read_from_lst(indir=indir, grdlst=grdlst, group=group)
        elif minlat!=None and Nlat!=None and minlon!=None and Nlon!=None and dlat!=None and dlon!=None:
            self._read_from_lat_lon(indir=indir, group=group, minlat=minlat, Nlat=Nlat, dlat=dlat, minlon=minlon, Nlon=Nlon, dlon=dlon)
        else:
            self._read_from_dir(indir=indir, group=group)
                
            
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
    
    def _read_from_lat_lon(self, indir, group, minlat, Nlat, dlat, minlon, Nlon, dlon):
        latArr=minlat+np.arange(Nlat)*dlat
        lonArr=minlon+np.arange(Nlon)*dlon
        for lat in latArr:
            for lon in lonArr:
                name='%g_%g' %(lon, lat)
                infname=indir+'/'+name+'_mod'
                if not os.path.isfile(infname):
                    warnings.warn('profile: lon='+str(lon)+' lat='+str(lat)+' not exists!', UserWarning, stacklevel=1)
                    continue
                inArr = np.loadtxt(infname)
                dset = group.create_dataset( name=name, shape=inArr.shape, data=inArr)
                dset.attrs.create(name = 'lon', data=lon, dtype='f')
                dset.attrs.create(name = 'lat', data=lat, dtype='f')
        return
    
    def _read_from_dir(self, indir, group):
        for infname in glob.glob(indir+'/*_mod'):
            f1=infname.split('/')[-1]
            f2=f1.split('_')
            lon = float(f2[0])
            lat = float(f2[1])
            name = '%g_%g' %(lon, lat)
            inArr = np.loadtxt(infname)
            dset = group.create_dataset( name=name, shape=inArr.shape, data=inArr)
            dset.attrs.create(name = 'lon', data=lon, dtype='f')
            dset.attrs.create(name = 'lat', data=lat, dtype='f')
        return
    
    def verticalExtend(self, inname, outname, block, refname ='ak135', dz=[], depthlst=[], z0lst =[],  maxdepth = 410.,
                    outdir = None, header={'depth': 0, 'vs':1, 'vp':2, 'rho':3, 'Qs':4}, dz2=10., dz3= 50. ):
        """
        Vertical profile extension
        Only suitable when input is upper mantle model( zmax < 410km)
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
        dz, depthlst - list for depth interval and bottom depth (grid)
        maxdepth     - maximum depth to extend (grid)
        outdir       - txt output directory( default = None)
        header       - header information
        dz2, dz3     - depth interval for step 2, 3
        ------------------------------------------------------------------------------------------------
        Output:
        
        =================================================================================================
        """
        # Generate numpy array for depth interpolation
        dz = np.asarray(dz)
        depthlst = np.asarray(depthlst)
        if len(dz) == 0 or len(depthlst) == 0 or len(dz) != len(depthlst) : 
            raise ValueError('Error input for depth!')
        if len(z0lst) != len(dz):
            if block ==True:
                print 'Note: Output is block model!'
                z0lst = np.ones(len(depthlst)) * dz / 2.
            else:
                print 'Note: Output is grid model!'
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
        if not block:
            depthArr=np.append(depthArr, depthlst[-1])
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
        else:
            step2 = False
        # determine whether to extend to maxdepth(>410km) or not
        if maxdepth > 410. and (maxdepth-410.)%dz3 ==0:
            step3 = True
            ak135Arr = self[refname][...]
            depthak135=ak135Arr[:,0]
            RhoAk135=ak135Arr[:,1]
            VpAk135=ak135Arr[:,2]
            VsAk135=ak135Arr[:,3]
            QAk135=ak135Arr[:,5]
            dzAttrs=np.append(dzAttrs, dz3)
            depthAttrs=np.append(depthAttrs, maxdepth)
        else:
            step3 = False
        # write header information
        for hkey in header.keys():
            group.attrs.create(name = hkey, data=header[hkey], dtype='i')
        group.attrs.create(name = 'dz', data=dzAttrs, dtype='f')
        group.attrs.create(name = 'depth', data=depthAttrs, dtype='f')
        if block:
            group.attrs.create(name = 'isblock', data=1, dtype='i')
        else:
            group.attrs.create(name = 'isblock', data=0, dtype='i')
        # loop over each profiles
        for index in self[inname].keys():
            depthInter = np.copy(depthArr)
            inArr = self[inname][index][...]
            depth=inArr[:,0]
            Vs=inArr[:,1]
            Vp=inArr[:,2]
            Rho=inArr[:,3]
            Q=inArr[:,4]
            # Interpolate to depthInter array for each profile
            VpInter=np.interp(depthInter, depth, Vp)
            VsInter=np.interp(depthInter, depth, Vs)
            RhoInter=np.interp(depthInter, depth, Rho)
            QInter=np.interp(depthInter, depth, Q)
            if step2:
                # Extend to ak135 410km/maxdepth with linear trend, with interval of dz1
                Vsak135_410=4.8702
                Vpak135_410=9.0302
                Rhoak135_410=3.5068
                Qak135_410=146.57
                if maxdepth > 410.:
                    if block:
                        depInter2=depthlst[-1]+np.arange((410. - depthlst[-1])/dz2)*dz2 + dz2/2. # depth array for interpolation
                    else:
                        depInter2=depthlst[-1]+np.arange((410. - depthlst[-1])/dz2)*dz2 + dz2 # depth array for interpolation
                else:
                    if block:
                        depInter2=depthlst[-1]+np.arange((maxdepth - depthlst[-1])/dz2)*dz2  + dz2/2.# depth array for interpolation
                    else:
                        depInter2=depthlst[-1]+np.arange((maxdepth - depthlst[-1])/dz2)*dz2 +dz2# depth array for interpolation
                Vsak135_2=np.append(Vs[-1], Vsak135_410)
                Vpak135_2=np.append(Vp[-1], Vpak135_410)
                Rhoak135_2=np.append(Rho[-1], Rhoak135_410)
                Qak135_2=np.append(Q[-1], Qak135_410)
                depthak135_2=np.append(depth[-1], 410.)
                
                VsInter2=np.interp(depInter2, depthak135_2, Vsak135_2)
                VpInter2=np.interp(depInter2, depthak135_2, Vpak135_2)
                RhoInter2=np.interp(depInter2, depthak135_2, Rhoak135_2)
                QInter2=np.interp(depInter2, depthak135_2, Qak135_2)
                
                depthInter=np.append(depthInter, depInter2)
                VsInter=np.append(VsInter,VsInter2)
                VpInter=np.append(VpInter,VpInter2)
                RhoInter=np.append(RhoInter,RhoInter2)
                QInter=np.append(QInter,QInter2)
            if step3:
                # Interpolate to maxdepth if it is larger than 410 km 
                Vsak135_3=VsAk135[(depthak135>410.)]
                Vpak135_3=VpAk135[(depthak135>410.)]
                Rhoak135_3=RhoAk135[(depthak135>410.)]
                Qak135_3=QAk135[(depthak135>410.)]
                depthak135_3=depthak135[(depthak135>410.)]
                if block:
                    depInter3=410.+np.arange((maxdepth-410.)/dz3)*dz3+dz3/2.
                else:
                    depInter3=410.+np.arange((maxdepth-410.)/dz3)*dz3+dz3
                # interpolation
                VsInter3=np.interp(depInter3, depthak135_3, Vsak135_3)
                VpInter3=np.interp(depInter3, depthak135_3, Vpak135_3)
                RhoInter3=np.interp(depInter3, depthak135_3, Rhoak135_3)
                QInter3=np.interp(depInter3, depthak135_3, Qak135_3)
                # Append to interpolated data
                depthInter=np.append(depthInter, depInter3)
                VsInter=np.append(VsInter,VsInter3)
                VpInter=np.append(VpInter,VpInter3)
                RhoInter=np.append(RhoInter,RhoInter3)
                QInter=np.append(QInter,QInter3)
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
            except RuntimeError:
                pass
        group.attrs.create(name = 'depthArr', data=depthInter, dtype='f')
        return
    
    def getavg(self, modelname ):
        depthArr = self[modelname].attrs['depthArr']
        Vs = np.ones(depthArr.size)
        Vp = np.ones(depthArr.size)
        Rho = np.ones(depthArr.size)
        Q = np.ones(depthArr.size)
        Nm = len(self[modelname].keys())
        for index in self[modelname].keys():
            inArr = self[modelname][index][...]
            Vs = Vs + inArr[:,1]
            Vp = Vp + inArr[:,2]
            Rho = Rho + inArr[:,3]
            Q = Q + inArr[:,4]
        Vs = Vs / Nm
        Vp = Vp / Nm
        Rho = Rho / Nm
        Q = Q / Nm
        outArr=np.append(depthArr, Vs)
        outArr=np.append(outArr, Vp)
        outArr=np.append(outArr, Rho)
        outArr=np.append(outArr, Q)
        outArr=outArr.reshape((5, depthArr.size))
        outArr=outArr.T
        self[modelname].attrs.create( name='avg_model', shape=outArr.shape, data=outArr)
        return
            
    def minmaxlonlat(self, modelname):
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
    
    def horizontalExtend(self, modelname, minlat, maxlat, dlat, minlon, maxlon, dlon,
                sfx='_ses3d', Dref=500., outdir = None):
        avgArr = self[modelname].attrs['avg_model'][...]
        latarr = minlat + np.arange( (maxlat-minlat)/dlat + 1)*dlat
        lonarr = minlon + np.arange( (maxlon-minlon)/dlon + 1)*dlon
        outname = modelname+sfx
        try:
            del self[outname]
        except:
            pass
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
                    if (abs(elat - lat) <=dlat and abs (elon-lon) == dlon) or (abs(elat - lat) ==dlat and abs (elon-lon) <= dlon):
                        break
                cname = '%g'%(clon) + '_%g' %(clat)
                cArr = self[outname][cname][...]
                if mindist < Dref:
                    outArr=npr.evaluate ( '((Dref-mindist)*cArr+mindist*avgArr)/Dref ')
                    # outArr=((Dref-mindist)*cArr+mindist*avgArr)/Dref 
                else:
                    outArr=avgArr
                dset = self[outname].create_dataset( name=name, shape=outArr.shape, data=outArr)
                dset.attrs.create(name = 'lon', data=lon, dtype='f')
                dset.attrs.create(name = 'lat', data=lat, dtype='f')
                if outdir !=None:
                # save to txt file
                    if not os.path.isdir(outdir):
                        os.makedirs(outdir)
                    np.savetxt(outdir+'/'+name+'_mod', outArr, fmt='%g')
        self[outname].attrs.create(name = 'minlat', data=minlat, dtype='f')
        self[outname].attrs.create(name = 'maxlat', data=maxlat, dtype='f')
        self[outname].attrs.create(name = 'dlat', data=dlat, dtype='f')
        self[outname].attrs.create(name = 'minlon', data=minlon, dtype='f')
        self[outname].attrs.create(name = 'maxlon', data=maxlon, dtype='f')
        self[outname].attrs.create(name = 'dlon', data=dlon, dtype='f')
        print '================ End horizontal extrapolation =============='
        return 
#     
#     def horizontalExtendMP(self, modelname, minlat, maxlat, dlat, minlon, maxlon, dlon, outdir, 
#                     sfx='_ses3d', Dref=500., deleteflag = False):
#         avgArr = self[modelname].attrs['avg_model'][...]
#         latarr = minlat + np.arange( (maxlat-minlat)/dlat + 1)*dlat
#         lonarr = minlon + np.arange( (maxlon-minlon)/dlon + 1)*dlon
#         outname = modelname+sfx
#         try:
#             del self[outname]
#         except:
#             pass
#         self.copy( source = modelname, dest = outname )
#         positionArr=[]
#         for lon in lonarr[:2]:
#             for lat in latarr[:2]:
#                 name='%g'%(lon) + '_%g' %(lat)
#                 print name
#                 if (name in self[modelname].keys()): continue
#                 positionArr.append(np.array([lon, lat]))
#         # latlon_lst= []
#         # for index in self[outname].keys():
#         #     elat = self[outname][index].attrs['lat']
#         #     elon = self[outname][index].attrs['lon']
#         #     name = '%g'%(elon) + '_%g' %(elat)
#         #     latlon_lst.append([elat, elon])
#         # for index in dataset.keys():
#         #     elat = dataset[index].attrs['lat']
#         #     elon = dataset[index].attrs['lon']
#         #     print elat, lon
#         # return
#         #     dist=great_circle((elat, elon),(lat, lon)).km
#         #     # dist, az, baz=obsGeo.gps2DistAzimuth(geopoint.lat, geopoint.lon, Nlat, Nlon ) # distance is in m
#         #     # dist=dist/1000
#         #     if dist<mindist:
#         #         mindist=dist
#         #         clon=elon
#         #         clat=elat
#         # cname = '%g'%(clon) + '_%g' %(clat)
#         dset = self[modelname]
#         print '================ Start horizontal extrapolation (MP) =============='
#         HEXTRAPOLATE = partial(PositionHExtrapolate, avgArr=avgArr,
#                     outdir=outdir, dataset=dset, Dref=Dref)
#         pool =multiprocessing.Pool()
#         pool.map(HEXTRAPOLATE, positionArr) #make our results with a map call
#         pool.close() #we are not adding any more processes
#         pool.join() #tell it to wait until all threads are done before going on
#         # read txt model into data set
#         return
#         for position in positionArr:
#             lon=position[0]
#             lat=position[1]
#             name='%g'%(lon) + '_%g' %(lat)
#             inArr = np.loadtxt(outdir+'/'+name+'_mod')
#             dset = self[modelname].create_dataset( name=name, shape=inArr.shape, data=inArr)
#             dset.attrs.create(name = 'lon', data=lon, dtype='f')
#             dset.attrs.create(name = 'lat', data=lat, dtype='f')
#         if deleteflag == True:
#             shutil.rmtree(outdir)
#         self[outname].attrs.create(name = 'minlat', data=minlat, dtype='f')
#         self[outname].attrs.create(name = 'maxlat', data=maxlat, dtype='f')
#         self[outname].attrs.create(name = 'dlat', data=dlat, dtype='f')
#         self[outname].attrs.create(name = 'minlon', data=minlon, dtype='f')
#         self[outname].attrs.create(name = 'maxlon', data=maxlon, dtype='f')
#         self[outname].attrs.create(name = 'dlon', data=dlon, dtype='f')
#         print '================ End horizontal extrapolation (MP) =============='
#         return 
#     
# 
# def PositionHExtrapolate( position, avgArr, outdir, dataset, Dref=500.):
#     lon=position[0]
#     lat=position[1]
#     mindist=99999
#     clon=0
#     clat=0
#     name='%g'%(lon) + '_%g' %(lat)
#     print 'Extending to ', name
#     # for index in dataset.keys():
#     #     elat = dataset[index].attrs['lat']
#     #     elon = dataset[index].attrs['lon']
#     #     print elat, lon
#     # return
#     #     dist=great_circle((elat, elon),(lat, lon)).km
#     #     # dist, az, baz=obsGeo.gps2DistAzimuth(geopoint.lat, geopoint.lon, Nlat, Nlon ) # distance is in m
#     #     # dist=dist/1000
#     #     if dist<mindist:
#     #         mindist=dist
#     #         clon=elon
#     #         clat=elat
#     # cname = '%g'%(clon) + '_%g' %(clat)
#     # cArr = dataset[cname][...]
#     # if mindist < Dref:
#     #     outArr=((Dref-mindist)*cArr+mindist*avgArr)/Dref
#     # else:
#     #     outArr=avgArr
#     # # save to txt file
#     # if not os.path.isdir(outdir):
#     #     os.makedirs(outdir)
#     # np.savetxt(outdir+'/'+name+'_mod', outArr, fmt='%g')
#     # return
#         
#         
    
