# -*- coding: utf-8 -*-
"""
A python module for SES3D block file manipulation.
Modified from python script in SES3D package( by Andreas Fichtner and Lion Krischer)
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import os
from lasif import rotations
from lasif import colors
import colormaps
from mpl_toolkits.basemap import Basemap, shiftgrid
import h5py
import obspy.geodetics.base


#########################################################################
#- define submodel model class
#########################################################################
class ses3d_submodel(object):
    """
    Class defining an ses3d submodel
    ===============================================================================
    Parameters:
    lat, lon, r            - latitude, longitude, radius array ( grid position)
    lat_rot, lon_rot       - rotated( or meshgrid ) latitude, longitude array
    dvsv, dvsh, dvp, drho  - Vsv, Vsh, Vp, density model ( block )
    ===============================================================================
    """
    def __init__(self):
        #- coordinate lines
        self.lat=np.zeros(1)
        self.lon=np.zeros(1)
        self.r=np.zeros(1)
        #- rotated coordinate lines
        self.lat_rot=np.zeros(1)
        self.lon_rot=np.zeros(1)
        #- field
        self.dvsv=np.zeros((1, 1, 1))
        self.dvsh=np.zeros((1, 1, 1))
        self.drho=np.zeros((1, 1, 1))
        self.dvp=np.zeros((1, 1, 1))
        
        
class ses3d_model(object):
    """
    An object for reading, writing, plotting and manipulating ses3d model
    ===========================================================================
    Parameters:
    nsubvol         - number of subvolumes
    projection      - plot type
    m               - list to store sub models
    phi             - rotation angle
    n               - rotation axis
    ===========================================================================
    """
    def __init__(self):
        """ initiate the ses3d_model class
        initiate list of submodels and read rotation_parameters.txt
        """
        self.nsubvol=0
        self.lat_min=0.0
        self.lat_max=0.0
        self.lon_min=0.0
        self.lon_max=0.0
        self.lat_centre=0.0
        self.lon_centre=0.0
        self.projection="global"
        self.m=[]
        #- read rotation parameters
        self.phi=0.0
        self.n = np.array([0., 1., 0.])
        return
    
    def copy(self):
        """ Copy a model
        """
        res=ses3d_model()
        res.nsubvol=self.nsubvol
        res.lat_min=self.lat_min
        res.lat_max=self.lat_max
        res.lon_min=self.lon_min
        res.lon_max=self.lon_max
        res.lat_centre=self.lat_centre
        res.lon_centre=self.lon_centre
        res.phi=self.phi
        res.n=self.n
        res.projection=self.projection
        res.d_lon=self.d_lon
        res.d_lat=self.d_lat
        for k in np.arange(self.nsubvol):
            subvol=ses3d_submodel()
            subvol.lat=self.m[k].lat
            subvol.lon=self.m[k].lon
            subvol.r=self.m[k].r
            subvol.lat_rot=self.m[k].lat_rot
            subvol.lon_rot=self.m[k].lon_rot
            subvol.v=self.m[k].v
            res.m.append(subvol)
        return res

    def __rmul__(self, factor):
        """ override left-multiplication of an ses3d model by a scalar factor
        """
        res=ses3d_model()
        res.nsubvol=self.nsubvol
        res.lat_min=self.lat_min
        res.lat_max=self.lat_max
        res.lon_min=self.lon_min
        res.lon_max=self.lon_max
        res.lat_centre=self.lat_centre
        res.lon_centre=self.lon_centre
        res.phi=self.phi
        res.n=self.n
        res.projection=self.projection
        res.d_lon=self.d_lon
        res.d_lat=self.d_lat
        for k in np.arange(self.nsubvol):
            subvol=ses3d_submodel()
            subvol.lat=self.m[k].lat
            subvol.lon=self.m[k].lon
            subvol.r=self.m[k].r
            subvol.lat_rot=self.m[k].lat_rot
            subvol.lon_rot=self.m[k].lon_rot
            subvol.v=factor*self.m[k].v
            res.m.append(subvol)
        return res
    
    def __add__(self,other_model):
        """ override addition of two ses3d models
        """
        res=ses3d_model()
        res.nsubvol=self.nsubvol
        res.lat_min=self.lat_min
        res.lat_max=self.lat_max
        res.lon_min=self.lon_min
        res.lon_max=self.lon_max
        res.lat_centre=self.lat_centre
        res.lon_centre=self.lon_centre
        res.phi=self.phi
        res.n=self.n
        res.projection=self.projection
        res.d_lon=self.d_lon
        res.d_lat=self.d_lat
        for k in np.arange(self.nsubvol):
            subvol=ses3d_submodel()
            subvol.lat=self.m[k].lat
            subvol.lon=self.m[k].lon
            subvol.r=self.m[k].r
            subvol.lat_rot=self.m[k].lat_rot
            subvol.lon_rot=self.m[k].lon_rot
            subvol.v=self.m[k].v+other_model.m[k].v
            res.m.append(subvol)
        return res
    
    def read_block(self, directory, verbose=False):
        """ read ses3d block files from a directory
        """
        #- read block files ====================================================
        fid_x=open(directory+'/block_x','r')
        fid_y=open(directory+'/block_y','r')
        fid_z=open(directory+'/block_z','r')
        if verbose==True:
            print 'read block files:'
            print '\t '+directory+'/block_x'
            print '\t '+directory+'/block_y'
            print '\t '+directory+'/block_z'
        ###
        # dx, dy, dz : x, y, z data in block_x/y/z
        ###
        dx=np.array(fid_x.read().strip().split('\n'),dtype=float)
        dy=np.array(fid_y.read().strip().split('\n'),dtype=float)
        dz=np.array(fid_z.read().strip().split('\n'),dtype=float)
        fid_x.close()
        fid_y.close()
        fid_z.close()
        #- read coordinate lines ===============================================
        self.nsubvol=int(dx[0])
        if verbose==True:
            print 'number of subvolumes: '+str(self.nsubvol)
        ###
        # idx, idy, idz : index for the first element in each subvolume
        ###
        idx=np.ones(self.nsubvol, dtype=int)
        idy=np.ones(self.nsubvol, dtype=int)
        idz=np.ones(self.nsubvol, dtype=int)
        for k in np.arange(1, self.nsubvol, dtype=int):
            idx[k]=int( dx[idx[k-1]] )+idx[k-1]+1
            idy[k]=int( dy[idy[k-1]] )+idy[k-1]+1
            idz[k]=int( dz[idz[k-1]] )+idz[k-1]+1
        for k in np.arange(self.nsubvol, dtype=int):
            subvol=ses3d_submodel()
            subvol.lat=90.0-dx[(idx[k]+1):(idx[k]+1+dx[idx[k]])]
            subvol.lon=dy[(idy[k]+1):(idy[k]+1+dy[idy[k]])]
            subvol.r  =dz[(idz[k]+1):(idz[k]+1+dz[idz[k]])]
            self.m.append(subvol)
        #- compute rotated version of the coordinate lines ====================
        if self.phi!=0.0:
            for k in np.arange(self.nsubvol, dtype=int):
                self.m[k].lat_rot, self.m[k].lon_rot \
                          = rotations.rotate_lat_lon(self.m[k].lat, self.m[k].lon, self.n, self.phi)
        else:
            for k in np.arange(self.nsubvol,dtype=int):
                self.m[k].lat_rot, self.m[k].lon_rot = np.meshgrid(self.m[k].lat, self.m[k].lon) 
                self.m[k].lat_rot = self.m[k].lat_rot.T
                self.m[k].lon_rot = self.m[k].lon_rot.T
        #- decide on global or regional model =======================================
        self.lat_min=90.0
        self.lat_max=-90.0
        self.lon_min=180.0
        self.lon_max=-180.0
        for k in np.arange(self.nsubvol):
            if np.min(self.m[k].lat_rot) < self.lat_min: self.lat_min = np.min(self.m[k].lat_rot)
            if np.max(self.m[k].lat_rot) > self.lat_max: self.lat_max = np.max(self.m[k].lat_rot)
            if np.min(self.m[k].lon_rot) < self.lon_min: self.lon_min = np.min(self.m[k].lon_rot)
            if np.max(self.m[k].lon_rot) > self.lon_max: self.lon_max = np.max(self.m[k].lon_rot)
        if ((self.lat_max-self.lat_min) > 90.0 or (self.lon_max-self.lon_min) > 90.0):
            self.projection = "global"
            self.lat_centre = (self.lat_max+self.lat_min)/2.0
            self.lon_centre = (self.lon_max+self.lon_min)/2.0
        else:
            self.projection = "regional"
            self.d_lat=5.0
            self.d_lon=5.0
        return
    
    def write_block(self, directory, verbose=False):
        """ write block files to a directory
        """
        #- write block files ====================================================
        fid_x=open(directory+'/block_x','w')
        fid_y=open(directory+'/block_y','w')
        fid_z=open(directory+'/block_z','w')
        if verbose==True:
            print 'write block files:'
            print '\t '+directory+'/block_x'
            print '\t '+directory+'/block_y'
            print '\t '+directory+'/block_z'
        ###
        # dx, dy, dz : x, y, z data in block_x/y/z
        ###
        fid_x.write(str(self.nsubvol)+'\n')
        fid_y.write(str(self.nsubvol)+'\n')
        fid_z.write(str(self.nsubvol)+'\n')
        for k in xrange(self.nsubvol):
            subvol=self.m[k]
            fid_x.write(str(int(subvol.lat.size))+'\n')
            fid_y.write(str(int(subvol.lon.size))+'\n')
            fid_z.write(str(int(subvol.r.size))+'\n')
            for i in xrange(subvol.lat.size):
                fid_x.write(str(float(90. - subvol.lat[i]))+'\n')
            for i in xrange(subvol.lon.size):
                fid_y.write(str(float(subvol.lon[i]))+'\n')
            for i in xrange(subvol.r.size):
                fid_z.write(str(float(subvol.r[i]))+'\n')
        fid_x.close()
        fid_y.close()
        fid_z.close()
        return
    
    def read_model(self, directory, filename, verbose=False):
        """ read a single ses3d model file from a directory, need block information
        ==============================================================
        Input format:
        No. of subvolume
        total No. for subvolume 1
        ... (data in nx, ny, nz)
        total No. for subvolume 2
        ...
        ==============================================================
        """
        #- read model volume ==================================================
        fid_m=open(directory+'/'+filename,'r')
        if verbose==True:
            print 'read model file: '+directory+'/'+filename
        v=np.array(fid_m.read().strip().split('\n'),dtype=float)
        fid_m.close()
        #- assign values ======================================================
        idx=1
        for k in np.arange(self.nsubvol):
            n = int(v[idx])
            nx = len(self.m[k].lat)-1
            ny = len(self.m[k].lon)-1
            nz = len(self.m[k].r)-1
            if filename=='dvsv':
                self.m[k].dvsv = v[(idx+1):(idx+1+n)].reshape(nx, ny, nz)
            if filename=='dvsh':
                self.m[k].dvsh = v[(idx+1):(idx+1+n)].reshape(nx, ny, nz)
            if filename=='drho':
                self.m[k].drho = v[(idx+1):(idx+1+n)].reshape(nx, ny, nz)
            if filename=='dvp':
                self.m[k].dvp = v[(idx+1):(idx+1+n)].reshape(nx, ny, nz)
            idx = idx+n+1
        return
    
    def write_model(self, directory, filename, verbose=False):
        """ write a single ses3d model file to a directory, need block information
        ==============================================================
        Output format:
        No. of subvolume
        total No. for subvolume 1
        ... (data in nx, ny, nz)
        total No. for subvolume 2
        ...
        ==============================================================
        """
        if not os.path.isdir(directory):
            os.makedirs(directory)
        with open(directory+'/'+filename,'w') as fid_m:
            if verbose==True:
                print 'write to file '+directory+'/'+filename
            fid_m.write(str(self.nsubvol)+'\n')
            for k in np.arange(self.nsubvol):
                nx=len(self.m[k].lat)-1
                ny=len(self.m[k].lon)-1
                nz=len(self.m[k].r)-1
                fid_m.write(str(nx*ny*nz)+'\n')
                if filename=='dvsv':
                    v = self.m[k].dvsv 
                if filename=='dvsh':
                    v = self.m[k].dvsh 
                if filename=='drho':
                    v = self.m[k].drho 
                if filename=='dvp':
                    v = self.m[k].dvp 
                for idx in np.arange(nx):
                    for idy in np.arange(ny):
                        for idz in np.arange(nz):
                            fid_m.write(str(v[idx,idy,idz])+'\n')
        return
    
    def read(self, directory, verbose=True):
        """ read ses3d model from a directory
        """
        print '===================== Read ses3d block model ====================='
        self.read_block(directory=directory, verbose=verbose)
        self.read_model(directory=directory, filename='dvsv', verbose=verbose)
        self.read_model(directory=directory, filename='dvsh', verbose=verbose)
        self.read_model(directory=directory, filename='drho', verbose=verbose)
        self.read_model(directory=directory, filename='dvp', verbose=verbose)
        print '=================================================================='
        return
    
    def write(self, directory, verbose=True):
        """ write ses3d model to a directory
        """
        if not os.path.isdir(directory):
            os.makedirs(directory)
        print '===================== Write ses3d block model ====================='
        self.write_block(directory=directory, verbose=verbose)
        self.write_model(directory=directory, filename='dvsv', verbose=verbose)
        self.write_model(directory=directory, filename='dvsh', verbose=verbose)
        self.write_model(directory=directory, filename='drho', verbose=verbose)
        self.write_model(directory=directory, filename='dvp', verbose=verbose)
        print '==================================================================='
        return
    
    def readh5model(self, infname, groupname, minlat=-999, maxlat=999, minlon=-999, maxlon=999, maxdepth=None ):
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
        MDataset = h5py.File(infname)
        # get latitude/longitude information
        if minlat < MDataset[groupname].attrs['minlat']: minlat = MDataset[groupname].attrs['minlat']
        if maxlat > MDataset[groupname].attrs['maxlat']: maxlat = MDataset[groupname].attrs['maxlat']
        if minlon < MDataset[groupname].attrs['minlon']: minlon = MDataset[groupname].attrs['minlon']
        if maxlon > MDataset[groupname].attrs['maxlon']: maxlon = MDataset[groupname].attrs['maxlon']
        dlon = MDataset[groupname].attrs['dlon']
        dlat = MDataset[groupname].attrs['dlat']
        self.lat_min=minlat-dlat/2.
        self.lat_max=maxlat+dlat/2.
        self.lon_min=minlon-dlon/2.
        self.lon_max=maxlon+dlon/2.
        # determine number of subvolumes, max depth and whether to interpolate or not 
        if maxdepth == None:
            dz = MDataset[groupname].attrs['dz']
            depth = MDataset[groupname].attrs['depth']
            depthArr = MDataset[groupname].attrs['depthArr']
            self.nsubvol = dz.size
        else:
            dz = MDataset[groupname].attrs['dz']
            depth = MDataset[groupname].attrs['depth']
            depthArr = MDataset[groupname].attrs['depthArr']
            if maxdepth > depth[-1]:
                raise ValueError('maximum depth is too large!')
            depth = depth [ np.where(depth<maxdepth)[0] ]
            bdz = dz[depth.size]
            if depth.size == 0 and maxdepth % bdz != 0:
                maxdepth = int( maxdepth / bdz ) * bdz
                print 'actual max depth:', maxdepth, 'km'
            elif ( maxdepth - depth[-1]) % bdz !=0:
                maxdepth = int( ( maxdepth - depth[-1]) / bdz ) * bdz + depth[-1]
                print 'actual max depth:', maxdepth, 'km'
            depth = np.append(depth, maxdepth)
            dz = dz[:depth.size]
            self.nsubvol = depth.size
        ##############################
        # Get block information
        ##############################
        radius=6371.
        dx=dlat
        dy=dlon
        xmin = 90. - self.lat_max
        xmax = 90. - self.lat_min
        ymin = self.lon_min
        ymax = self.lon_max
        mindepth=0
        nzArr = np.array([], dtype=int)
        x0=xmin+dx/2.
        y0=ymin+dy/2.
        xG0=xmin
        yG0=ymin
        nx=int((xmax-xmin)/dx)
        ny=int((ymax-ymin)/dy)
        nGx=int((xmax-xmin)/dx)+1
        nGy=int((ymax-ymin)/dy)+1
        xArr=x0+np.arange(nx)*dx
        yArr=y0+np.arange(ny)*dy
        xGArr=xG0+np.arange(nGx)*dx
        yGArr=yG0+np.arange(nGy)*dy
        for k in xrange(self.nsubvol):
            rG0=radius-depth[k]
            zG0=depth[k]
            if k == 0:
                nzArr = np.append (nzArr, int((depth[k]-mindepth)/dz[k]) )
                nGz=int((depth[k]-mindepth)/dz[k])+1
            else:
                nzArr = np.append (nzArr, int((depth[k]-depth[k-1])/dz[k]) )
                nGz=int((depth[k]-depth[k-1])/dz[k])+1
            zGArr=zG0-np.arange(nGz)*dz[k]
            rGArr=rG0+np.arange(nGz)*dz[k]
            new_subM = ses3d_submodel()
            new_subM.lat = 90. - xGArr
            new_subM.lon = yGArr
            new_subM.r = rGArr
            self.m.append(new_subM)
        ##############################
        # Get velocity model
        ##############################
        tz = 0
        bz = 0
        vsindex =  MDataset[groupname].attrs['vs']
        vpindex =  MDataset[groupname].attrs['vp']
        rhoindex =  MDataset[groupname].attrs['rho']
        if MDataset[groupname].attrs['isblock']:
            group = MDataset[groupname]
        else:
            try:
                group = MDataset[groupname+'_block']
            except:
                group = MDataset[groupname]
                warnings.warn('Input model is NOT block model! ', UserWarning, stacklevel=1)
        for k in xrange(self.nsubvol):
            self.m[k].dvsv = np.zeros((nx, ny, nzArr[k]))
            self.m[k].dvsh = np.zeros((nx, ny, nzArr[k]))
            self.m[k].dvp = np.zeros((nx, ny, nzArr[k]))
            self.m[k].drho = np.zeros((nx, ny, nzArr[k]))
            bz = bz + nzArr[k]
            for ix in xrange(nx):
                for iy in xrange(ny):
                    lat = 90. - xArr[ix]
                    lon = yArr [iy]
                    name='%g_%g' %(lon, lat)
                    depthProf = group[name][...]
                    self.m[k].dvsv[ix, iy, :] = (depthProf[tz:bz, vsindex])[::-1]
                    self.m[k].dvsh[ix, iy, :] = (depthProf[tz:bz, vsindex])[::-1]
                    self.m[k].dvp[ix, iy, :] = (depthProf[tz:bz, vpindex])[::-1]
                    self.m[k].drho[ix, iy, :] = (depthProf[tz:bz, rhoindex])[::-1]
            tz = tz + nzArr[k]
            self.m[k].lat_rot, self.m[k].lon_rot = np.meshgrid(self.m[k].lat, self.m[k].lon) 
            self.m[k].lat_rot = self.m[k].lat_rot.T
            self.m[k].lon_rot = self.m[k].lon_rot.T
        return
            
    def vsLimit(self, vsmin):
        """ Reassign the model value where vs < vsmin
        """
        vpmin=0.9409+2.0947*vsmin-0.8206*vsmin**2+0.2683*vsmin**3-0.0251*vsmin**4
        rhomin=1.6612*vpmin-0.4721*vpmin**2+0.0671*vpmin**3-0.0043*vpmin**4+0.000106*vpmin**5
        for k in xrange(self.nsubvol):
            indexS = self.m[k].dvsv < vsmin
            indexL = np.logical_not(indexS)
            self.m[k].dvsv = indexS *  vsmin + indexL * self.m[k].dvsv
            self.m[k].dvsh = indexS * vsmin + indexL * self.m[k].dvsh
            self.m[k].dvp = indexS * vpmin + indexL * self.m[k].dvp
            self.m[k].drho = indexS * rhomin + indexL * self.m[k].drho
        return
    
    def norm(self, modelname):
        """Compute the L2 norm
        """
        N=0.0
        #- Loop over subvolumes. ----------------------------------------------
        for n in np.arange(self.nsubvol):
            #- Size of the array.
            nx=len(self.m[n].lat)-1
            ny=len(self.m[n].lon)-1
            nz=len(self.m[n].r)-1
            #- Compute volume elements.
            if modelname == 'dvsv':
                v = self.m[n].dvsv 
            if modelname == 'dvsh':
                v = self.m[n].dvsh 
            if modelname == 'drho':
                v = self.m[n].drho 
            if modelname == 'dvp':
                v = self.m[n].dvp 
            dV=np.zeros(np.shape(v))
            theta=(90.0-self.m[n].lat)*np.pi/180.0
            dr=self.m[n].r[1]-self.m[n].r[0]
            dphi=(self.m[n].lon[1]-self.m[n].lon[0])*np.pi/180.0
            dtheta=theta[1]-theta[0]
            for idx in np.arange(nx):
                for idy in np.arange(ny):
                    for idz in np.arange(nz):
                        dV[idx,idy,idz]=theta[idx]*(self.m[n].r[idz])**2
            dV=dr*dtheta*dphi*dV
            #- Integrate.
            N+=np.sum(dV*(v)**2)
        #- Finish. ------------------------------------------------------------
        return np.sqrt(N)

    def _smooth_horizontal_single(self, sigma, modelname, filter_type='neighbour'):
        """
        Experimental function for smoothing in horizontal directions for given model type
        ========================================================================================
        Input parameters:
        sigma       - filter width ('gauss') or iterations ('neighbour')
        modelname   - modelname ('dvsv', 'dvsh', 'dvp', 'drho')
        filter_type - gauss (Gaussian smoothing), neighbour (average over neighbouring cells)
        ========================================================================================
        WARNING: Currently, the smoothing only works within each subvolume. The problem 
        of smoothing across subvolumes without having excessive algorithmic complexity 
        and with fast compute times, awaits resolution ... .
        """
        #- Loop over subvolumes.---------------------------------------------------
        for n in np.arange(self.nsubvol):
            if modelname == 'dvsv':
                v_filtered = self.m[n].dvsv 
            if modelname == 'dvsh':
                v_filtered = self.m[n].dvsh 
            if modelname == 'drho':
                v_filtered = self.m[n].drho 
            if modelname == 'dvp':
                v_filtered = self.m[n].dvp
            v = np.copy(v_filtered)
            #- Size of the array.
            nx=len(self.m[n].lat)-1
            ny=len(self.m[n].lon)-1
            nz=len(self.m[n].r)-1
            #- Gaussian smoothing. --------------------------------------------------
            if filter_type=='gauss':
                #- Estimate element width.
                r=np.mean(self.m[n].r)
                dx=r*np.pi*(self.m[n].lat[0]-self.m[n].lat[1])/180.0
                #- Colat and lon fields for the small Gaussian.
                dn=3*np.ceil(sigma/dx)
                nx_min=np.round(float(nx)/2.0)-dn
                nx_max=np.round(float(nx)/2.0)+dn
                ny_min=np.round(float(ny)/2.0)-dn
                ny_max=np.round(float(ny)/2.0)+dn
                lon,colat=np.meshgrid(self.m[n].lon[ny_min:ny_max],90.0-self.m[n].lat[nx_min:nx_max])
                colat=np.pi*colat/180.0
                lon=np.pi*lon/180.0
                #- Volume element.
                dy=r*np.pi*np.sin(colat)*(self.m[n].lon[1]-self.m[n].lon[0])/180.0
                dV=dx*dy
                #- Unit vector field.
                x=np.cos(lon)*np.sin(colat)
                y=np.sin(lon)*np.sin(colat)
                z=np.cos(colat)
                #- Make a Gaussian centred in the middle of the grid. -----------------
                i=np.round(float(nx)/2.0)-1
                j=np.round(float(ny)/2.0)-1
                colat_i=np.pi*(90.0-self.m[n].lat[i])/180.0
                lon_j=np.pi*self.m[n].lon[j]/180.0
                x_i=np.cos(lon_j)*np.sin(colat_i)
                y_j=np.sin(lon_j)*np.sin(colat_i)
                z_k=np.cos(colat_i)
                #- Compute the Gaussian.
                G=x*x_i+y*y_j+z*z_k
                G=G/np.max(np.abs(G))
                G=r*np.arccos(G)
                G=np.exp(-0.5*G**2/sigma**2)/(2.0*np.pi*sigma**2)
                #- Move the Gaussian across the field. --------------------------------
                for i in np.arange(dn+1,nx-dn-1):
                    for j in np.arange(dn+1,ny-dn-1):
                        for k in np.arange(nz):
                            v_filtered[i,j,k]=np.sum(v[i-dn:i+dn,j-dn:j+dn,k]*G*dV)
            #- Smoothing by averaging over neighbouring cells. ----------------------
            elif filter_type=='neighbour':
                for iteration in np.arange(int(sigma)):
                    for i in np.arange(1,nx-1):
                        for j in np.arange(1,ny-1):
                            v_filtered[i,j,:]=(v[i,j,:]+v[i+1,j,:]+v[i-1,j,:]+v[i,j+1,:]+v[i,j-1,:])/5.0
        return
        
    def smooth_horizontal(self, sigma, filter_type='neighbour'):
        """Experimental function for smoothing in horizontal directions for all model types
        """
        self._smooth_horizontal_single(sigma=sigma, modelname='dvsv', filter_type=filter_type)
        self._smooth_horizontal_single(sigma=sigma, modelname='dvsh', filter_type=filter_type)
        self._smooth_horizontal_single(sigma=sigma, modelname='dvp', filter_type=filter_type)
        self._smooth_horizontal_single(sigma=sigma, modelname='drho', filter_type=filter_type)
        return
    
    def smooth_horizontal_adaptive(self, modelname, sigma):
        """Apply horizontal smoothing with adaptive smoothing length.
        """
        #- Find maximum smoothing length. -------------------------------------
        sigma_max=[]
        for n in xrange(self.nsubvol):
            if modelname=='dvsv':
                v = sigma.m[n].dvsv 
            if modelname=='dvsh':
                v = sigma.m[n].dvsh 
            if modelname=='drho':
                v = sigma.m[n].drho 
            if modelname=='dvp':
                v = sigma.m[n].dvp 
            sigma_max.append(v.max())
        #- Loop over subvolumes.-----------------------------------------------
        for n in xrange(self.nsubvol):
            #- Size of the array.
            nx=len(self.m[n].lat)-1
            ny=len(self.m[n].lon)-1
            nz=len(self.m[n].r)-1
            #- Estimate element width.
            r=np.mean(self.m[n].r)
            dx=r*np.pi*(self.m[n].lat[0]-self.m[n].lat[1])/180.0
            #- Colat and lon fields for the small Gaussian.
            dn=2*np.round(sigma_max[n]/dx)
            nx_min=np.round(float(nx)/2.0)-dn
            nx_max=np.round(float(nx)/2.0)+dn
            ny_min=np.round(float(ny)/2.0)-dn
            ny_max=np.round(float(ny)/2.0)+dn
            lon,colat=np.meshgrid(self.m[n].lon[ny_min:ny_max], 90.0-self.m[n].lat[nx_min:nx_max],dtype=float)
            colat=np.pi*colat/180.0
            lon=np.pi*lon/180.0
            #- Volume element.
            dy=r*np.pi*np.sin(colat)*(self.m[n].lon[1]-self.m[n].lon[0])/180.0
            dV=dx*dy
            #- Unit vector field.
            x=np.cos(lon)*np.sin(colat)
            y=np.sin(lon)*np.sin(colat)
            z=np.cos(colat)
            #- Make a Gaussian centred in the middle of the grid. ---------------
            i=np.round(float(nx)/2.0)-1
            j=np.round(float(ny)/2.0)-1
            colat_i=np.pi*(90.0-self.m[n].lat[i])/180.0
            lon_j=np.pi*self.m[n].lon[j]/180.0
            x_i=np.cos(lon_j)*np.sin(colat_i)
            y_j=np.sin(lon_j)*np.sin(colat_i)
            z_k=np.cos(colat_i)
            #- Distance from the central point.
            G=x*x_i+y*y_j+z*z_k
            G=G/np.max(np.abs(G))
            G=r*np.arccos(G)
            #- Move the Gaussian across the field. ------------------------------
            if modelname == 'dvsv':
                v_filtered = self.m[n].dvsv
                sigmaV = sigma.m[n].dvsv 
            if modelname == 'dvsh':
                v_filtered = self.m[n].dvsh
                sigmaV = sigma.m[n].dvsh 
            if modelname == 'drho':
                v_filtered = self.m[n].drho
                sigmaV = sigma.m[n].drho 
            if modelname == 'dvp':
                v_filtered = self.m[n].dvp
                sigmaV = sigma.m[n].dvp 
            v = np.copy(v_filtered)
            for i in np.arange(dn+1,nx-dn-1):
                for j in np.arange(dn+1,ny-dn-1):
                    for k in np.arange(nz):
                        #- Compute the actual Gaussian.
                        s=sigmaV[i, j, k]
                        if (s>0):
                            GG=np.exp(-0.5*G**2/s**2)/(2.0*np.pi*s**2)
                            #- Apply filter.
                            v_filtered[i,j,k]=np.sum(v[i-dn:i+dn,j-dn:j+dn,k]*GG*dV)
        return

    def convert_to_vts(self, outdir, modelname, pfx='', verbose=False, unit=True):
        """ Convert ses3d model to vts format for plotting with Paraview, VisIt
        ========================================================================================
        Input parameters:
        outdir      - output directory
        modelname   - modelname ('dvsv', 'dvsh', 'dvp', 'drho')
        pfx         - prefix of output files
        unit        - output unit sphere(radius=1) or not
        ========================================================================================
        """
        if not os.path.isdir(outdir): os.makedirs(outdir)
        from tvtk.api import tvtk, write_data
        from mayavi import mlab
        if unit: Rref=6471.
        else: Rref=1.
        for n in xrange(self.nsubvol):
            theta=(90.0-self.m[n].lat[:-1])*np.pi/180.
            phi=(self.m[n].lon[:-1])*np.pi/180.
            radius=self.m[n].r[:-1]
            theta, phi, radius = np.meshgrid(theta, phi, radius, indexing='ij')
            x = radius * np.sin(theta) * np.cos(phi)/Rref
            y = radius * np.sin(theta) * np.sin(phi)/Rref
            z = radius * np.cos(theta)/Rref
            dims = self.m[n].dvsv.shape
            pts = np.empty(z.shape + (3,), dtype=float)
            pts[..., 0] = x; pts[..., 1] = y; pts[..., 2] = z
            # Reorder the points, scalars and vectors,
            # so this is as per VTK's requirement of x first, y next and z last.
            pts = pts.transpose(2, 1, 0, 3).copy()
            pts.shape = pts.size / 3, 3
            sgrid = tvtk.StructuredGrid(dimensions=dims, points=pts)
            if modelname == 'dvsv':  v = self.m[n].dvsv 
            if modelname == 'dvsh':  v = self.m[n].dvsh 
            if modelname == 'drho':  v = self.m[n].drho 
            if modelname == 'dvp':   v = self.m[n].dvp
            sgrid.point_data.scalars = (v).ravel(order='F')
            sgrid.point_data.scalars.name = modelname
            outfname=outdir+'/'+pfx+modelname+'_'+str(n)+'.vts'
            write_data(sgrid, outfname)
        #     if showfig:
        #         # Now visualize the data.
        #         d = mlab.pipeline.add_dataset(sgrid)
        #         gx = mlab.pipeline.grid_plane(d)
        #         gy = mlab.pipeline.grid_plane(d)
        #         gy.grid_plane.axis = 'y'
        #         gz = mlab.pipeline.grid_plane(d)
        #         gz.grid_plane.axis = 'z'
        #         iso = mlab.pipeline.iso_surface(d)
        #         iso.contour.maximum_contour = 75.0
        #         vec = mlab.pipeline.vectors(d)
        #         vec.glyph.mask_input_points = True
        #         vec.glyph.glyph.scale_factor = 1.5
        # mlab.show()
        return
    
    
    def convert_to_vtk(self, directory, modelname, filename, verbose=False):
        """ convert ses3d model to vtk format for plotting with Paraview, VisIt, ... .
        """
        Rref=6471.
        #- preparatory steps
        nx=np.zeros(self.nsubvol, dtype=int)
        ny=np.zeros(self.nsubvol, dtype=int)
        nz=np.zeros(self.nsubvol, dtype=int)
        N=0
        for n in xrange(self.nsubvol):
            nx[n]=len(self.m[n].lat)
            ny[n]=len(self.m[n].lon)
            nz[n]=len(self.m[n].r)
            N=N+nx[n]*ny[n]*nz[n]
        #- open file and write header
        fid=open(directory+'/'+filename,'w')
        if verbose==True:
            print 'write to file '+directory+filename
        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        #- write grid points
        fid.write('POINTS '+str(N)+' float\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing grid points for subvolume '+str(n)
            for i in np.arange(nx[n]):
                for j in np.arange(ny[n]):
                    for k in np.arange(nz[n]):
                        theta=90.0-self.m[n].lat[i]
                        phi=self.m[n].lon[j]
                        #- rotate coordinate system
                        if self.phi!=0.0:
                            lat_rot, lon_rot = rotations.rotate_lat_lon(90.-theta, phi, self.n, -self.phi)
                            theta = 90.0 - lar_rot
                            phi = lon_rot
                            # theta,phi=rotate_coordinates(self.n,-self.phi,theta,phi)
                            #- transform to cartesian coordinates and write to file
                        theta=theta*np.pi/180.0
                        phi=phi*np.pi/180.0
                        r=self.m[n].r[k]/Rref ### !!!
                        x=r*np.sin(theta)*np.cos(phi)
                        y=r*np.sin(theta)*np.sin(phi)
                        z=r*np.cos(theta)
                        fid.write(str(x)+' '+str(y)+' '+str(z)+'\n')
        #- write connectivity
        n_cells=0
        for n in np.arange(self.nsubvol):
            n_cells=n_cells+(nx[n]-1)*(ny[n]-1)*(nz[n]-1)
        fid.write('\n')
        fid.write('CELLS '+str(n_cells)+' '+str(9*n_cells)+'\n')
        count=0
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing conectivity for subvolume '+str(n)
            for i in np.arange(1,nx[n]):
                for j in np.arange(1,ny[n]):
                    for k in np.arange(1,nz[n]):
                        a=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        b=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n] 
                        c=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        d=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]
                        e=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]-1
                        f=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]
                        g=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]-1
                        h=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]
                        fid.write('8 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)+' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')
            count=count+nx[n]*ny[n]*nz[n]
        #- write cell types
        fid.write('\n')
        fid.write('CELL_TYPES '+str(n_cells)+'\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing cell types for subvolume '+str(n)
            for i in np.arange(nx[n]-1):
                for j in np.arange(ny[n]-1):
                    for k in np.arange(nz[n]-1):
                        fid.write('11\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS scalars float\n')
        fid.write('LOOKUP_TABLE mytable\n')
        for n in np.arange(self.nsubvol):
            if verbose: print 'writing data for subvolume '+str(n)
            idx=np.arange(nx[n])
            idx[nx[n]-1]=nx[n]-2
            idy=np.arange(ny[n])
            idy[ny[n]-1]=ny[n]-2
            idz=np.arange(nz[n])
            idz[nz[n]-1]=nz[n]-2
            if modelname == 'dvsv':  v = self.m[n].dvsv 
            if modelname == 'dvsh':  v = self.m[n].dvsh 
            if modelname == 'drho':  v = self.m[n].drho 
            if modelname == 'dvp':   v = self.m[n].dvp
            for i in idx:
                for j in idy:
                    for k in idz:
                        fid.write(str(v[i,j,k])+'\n')
        #- clean up
        fid.close()
        return
    
    def convert_to_vtk_depth(self, depth, directory, filename, verbose=False):
        """ convert ses3d model to vtk format for plotting with Paraview, VisIt, ... .
        """
        Rref=6471.
        #- preparatory steps
        nx=np.zeros(self.nsubvol, dtype=int)
        ny=np.zeros(self.nsubvol, dtype=int)
        nz=np.zeros(self.nsubvol, dtype=int)
        N=0
        for n in np.arange(self.nsubvol):
            nx[n]=len(self.m[n].lat)
            ny[n]=len(self.m[n].lon)
            nz[n]=len(self.m[n].r)
            N=N+nx[n]*ny[n]*nz[n]
        #- open file and write header
        fid=open(directory+'/'+filename,'w')
        if verbose==True:
            print 'write to file '+directory+filename
        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        #- write grid points
        fid.write('POINTS '+str(N)+' float\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing grid points for subvolume '+str(n)
            for i in np.arange(nx[n]):
                for j in np.arange(ny[n]):
                    for k in np.arange(nz[n]):
                        theta=90.0-self.m[n].lat[i]
                        phi=self.m[n].lon[j]
                        #- rotate coordinate system
                        if self.phi!=0.0:
                            lat_rot, lon_rot = rotations.rotate_lat_lon(90.-theta, phi, self.n, -self.phi)
                            theta = 90.0 - lar_rot
                            phi = lon_rot
                            # theta,phi=rotate_coordinates(self.n,-self.phi,theta,phi)
                            #- transform to cartesian coordinates and write to file
                        theta=theta*np.pi/180.0
                        phi=phi*np.pi/180.0
                        r=self.m[n].r[k]/Rref ### !!!
                        x=r*np.sin(theta)*np.cos(phi)
                        y=r*np.sin(theta)*np.sin(phi)
                        z=r*np.cos(theta)
                        fid.write(str(x)+' '+str(y)+' '+str(z)+'\n')
        #- write connectivity
        n_cells=0
        for n in np.arange(self.nsubvol):
            n_cells=n_cells+(nx[n]-1)*(ny[n]-1)*(nz[n]-1)
        fid.write('\n')
        fid.write('CELLS '+str(n_cells)+' '+str(9*n_cells)+'\n')
        count=0
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing conectivity for subvolume '+str(n)
            for i in np.arange(1,nx[n]):
                for j in np.arange(1,ny[n]):
                    for k in np.arange(1,nz[n]):
                        a=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        b=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n] 
                        c=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        d=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]
                        e=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]-1
                        f=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]
                        g=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]-1
                        h=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]
                        fid.write('8 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)+' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')
            count=count+nx[n]*ny[n]*nz[n]
        #- write cell types
        fid.write('\n')
        fid.write('CELL_TYPES '+str(n_cells)+'\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing cell types for subvolume '+str(n)
            for i in np.arange(nx[n]-1):
                for j in np.arange(ny[n]-1):
                    for k in np.arange(nz[n]-1):
                        fid.write('11\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS scalars float\n')
        fid.write('LOOKUP_TABLE mytable\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing data for subvolume '+str(n)
            idx=np.arange(nx[n])
            idx[nx[n]-1]=nx[n]-2
            idy=np.arange(ny[n])
            idy[ny[n]-1]=ny[n]-2
            idz=np.arange(nz[n])
            idz[nz[n]-1]=nz[n]-2
            if modelname =='dvsv':
                v = self.m[n].dvsv 
            if modelname =='dvsh':
                v = self.m[n].dvsh 
            if modelname =='drho':
                v = self.m[n].drho 
            if modelname =='dvp':
                v = self.m[n].dvp
            for i in idx:
                for j in idy:
                    for k in idz:
                        fid.write(str(v[i,j,k])+'\n')
        #- clean up
        fid.close()
        return
    
    def plot_slice(self, depth, modelname, min_val_plot=None, max_val_plot=None, colormap='tomo_80_perc_linear_lightness',
                   resolution='i', save_under=None, verbose=False, mapfactor=2, geopolygons=None):
        """
        Plot horizontal slices through an ses3d model
        ===================================================================================================
        Input parameters:
        depth        - depth to be plotted, if given value is N/A, closest value will be chosen
        modelname    - modelname ('dvsv', 'dvsh', 'dvp', 'drho')
        min_val_plot - vmin/vmax for the colorbar
        max_val_plot -
        colormap     - colormap to be used
        resolution   - resolution for plotting
        save_under   - save figure as *.png with the filename "save_under". Prevents plotting of the slice
        mapfactor    - map factor for zoom for regional plotting with orthographic projection
        geopolygons  - geological polygons to plot on the map
        ===================================================================================================
        Note:
        The model is actually a block model, namely, the model values are NOT assigned to grid point.
        However, as an approximation, we can assume each grid has the value corresponds to the nearest block. 
        """
        radius=6371.0-depth
        # ax=plt.subplot(111)
        #- set up a map and colourmap -----------------------------------------
        if self.projection=='merc':
            m=Basemap(projection='merc', llcrnrlat=self.lat_min, urcrnrlat=self.lat_max, llcrnrlon=self.lon_min,
                      urcrnrlon=self.lon_max, lat_ts=20, resolution=resolution)
            m.drawparallels(np.arange(self.lat_min,self.lat_max,self.d_lon), labels=[1,0,0,1])
            m.drawmeridians(np.arange(self.lon_min,self.lon_max,self.d_lat), labels=[1,0,0,1])
        
        elif self.projection=='global':
            self.lat_centre = (self.lat_max+self.lat_min)/2.0
            self.lon_centre = (self.lon_max+self.lon_min)/2.0
            m=Basemap(projection='ortho',lon_0=self.lon_centre, lat_0=self.lat_centre, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        
        elif self.projection=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=self.lon_min, lat_0=self.lat_min, resolution='l')
            m = Basemap(projection='ortho', lon_0=self.lon_min,lat_0=self.lat_min, resolution=resolution,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif self.projection=='lambert':
            self.lat_centre = (self.lat_max+self.lat_min)/2.0
            self.lon_centre = (self.lon_max+self.lon_min)/2.0
            distEW, az, baz=obspy.geodetics.base.gps2dist_azimuth(self.lat_min, self.lon_min,
                                self.lat_min, self.lon_max) # distance is in m
            distNS, az, baz=obspy.geodetics.base.gps2dist_azimuth(self.lat_min, self.lon_min,
                                self.lat_max+1.7, self.lon_min) # distance is in m
            m = Basemap(width=distEW, height=distNS,
            rsphere=(6378137.00,6356752.3142),\
            resolution='l', projection='lcc',\
            lat_1=self.lat_min, lat_2=self.lat_max, lat_0=self.lat_centre+1.2, lon_0=self.lon_centre)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=2, dashes=[2,2], labels=[1,0,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=2, dashes=[2,2], labels=[0,0,1,1], fontsize=15)
            
        
        
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries()
        # m.drawmapboundary(fill_color=[1.0,1.0,1.0])
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        # m.drawlsmask(land_color='0.8', ocean_color='#99ffff')
        m.drawmapboundary(fill_color="white")
        # plt.show()
        # return
        cmap = colors.get_colormap(colormap)
        # if colormap=='tomo':
        #     my_colormap=colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0],\
        #         0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], \
        #         0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        # elif colormap=='mono':
        #     my_colormap=colormaps.make_colormap({0.0:[1.0,1.0,1.0], 0.15:[1.0,1.0,1.0], 0.85:[0.0,0.0,0.0], 1.0:[0.0,0.0,0.0]})
            
        #- loop over subvolumes to collect information ------------------------
        x_list=[]
        y_list=[]
        idz_list=[]
        N_list=[]
        for k in np.arange(self.nsubvol):
            nx=len(self.m[k].lat)
            ny=len(self.m[k].lon)
            r=self.m[k].r
            #- collect subvolumes within target depth
            if (r.max()>=radius) & (min(r)<radius):
                N_list.append(k)
                r=r[0:len(r)-1]
                idz=min( np.where(min(np.abs(r-radius))==np.abs(r-radius))[0])
                if idz==len(r): idz-=idz
                idz_list.append(idz)
                if verbose:
                    print 'true plotting depth: '+str(6371.0-r[idz])+' km'
                x, y=m(self.m[k].lon_rot[0:nx-1,0:ny-1], self.m[k].lat_rot[0:nx-1,0:ny-1]) # approximately, note that the model is actualy a block model
                x_list.append(x)
                y_list.append(y)
        #- make a (hopefully) intelligent colour scale ------------------------
        if min_val_plot is None:
            if len(N_list)>0:
                #- compute some diagnostics
                min_list=[]
                max_list=[]
                percentile_list=[]
                for k in np.arange(len(N_list)):
                    if modelname=='dvsv':
                        v = self.m[N_list[k]].dvsv 
                    if modelname=='dvsh':
                        v = self.m[N_list[k]].dvsh 
                    if modelname=='drho':
                        v = self.m[N_list[k]].drho 
                    if modelname=='dvp':
                        v = self.m[N_list[k]].dvp
                    min_list.append(np.min(v[:,:,idz_list[k]]))
                    max_list.append(np.max(v[:,:,idz_list[k]]))
                    percentile_list.append(np.percentile(np.abs(v[:,:,idz_list[k]]), 99.0))
                minval=np.min(min_list)
                maxval=np.max(max_list)
                percent=np.max(percentile_list)
                #- min and max roughly centred around zero
                if (minval*maxval<0.0):
                    max_val_plot=percent
                    min_val_plot=-max_val_plot
                #- min and max not centred around zero
                else:
                    max_val_plot=maxval
                    min_val_plot=minval
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        #- loop over subvolumes to plot ---------------------------------------
        for k in np.arange(len(N_list)):
            if modelname=='dvsv':
                v = self.m[N_list[k]].dvsv 
            if modelname=='dvsh':
                v = self.m[N_list[k]].dvsh 
            if modelname=='drho':
                v = self.m[N_list[k]].drho 
            if modelname=='dvp':
                v = self.m[N_list[k]].dvp
            im=m.pcolormesh(x_list[k], y_list[k], v[:,:,idz_list[k]],
                    shading='gouraud', cmap=cmap, vmin=min_val_plot, vmax=max_val_plot)
        #- make a colorbar and title ------------------------------------------
        cb=m.colorbar(im,"right", size="3%", pad='2%', ticks=np.arange( (max_val_plot-min_val_plot)/0.1+1)*0.1+min_val_plot )
        cb.ax.tick_params(labelsize=15)
        if modelname == 'drho':
            cb.set_label('g/cm^3', fontsize=20, rotation=90)
        else:
            cb.set_label('km/sec', fontsize=20, rotation=90)
        # im.ax.tick_params(labelsize=20)
        # plt.title(modelname+ ' at ' + str(depth)+' km', fontsize=30)
        #- save image if wanted -----------------------------------------------
        if save_under is None:
            plt.show()
        else:
            plt.savefig(save_under+'.png', format='png', dpi=200)
            plt.close()
        return
    
    def plot_threshold(self, val, modelname, min_val_plot, max_val_plot, colormap='afmhot_r', resolution='i', verbose=False, geopolygons=None):
        """
        Plot depth to a certain threshold value 'val' in an ses3d model
        ==================================================================================================
        Input parameters:
        val          - threshold value
        modelname    - modelname ('dvsv', 'dvsh', 'dvp', 'drho')
        min_val_plot - vmin/vmax for the colorbar
        max_val_plot
        colormap     - colormap to be used
        resolution   - resolution for plotting
        save_under   - save figure as *.png with the filename "save_under". Prevents plotting of the slice
        geopolygons  - geological polygons to plot on the map
        ==================================================================================================
        val=threshold value
        min_val_plot, max_val_plot=minimum and maximum values of the colour scale
        colormap='tomo','mono'
        """
        #- set up a map and colourmap
        if self.projection=='regional':
            m=Basemap(projection='merc',llcrnrlat=self.lat_min,urcrnrlat=self.lat_max,\
                    llcrnrlon=self.lon_min,urcrnrlon=self.lon_max,lat_ts=20,resolution=resolution)
            m.drawparallels(np.arange(self.lat_min,self.lat_max,self.d_lon),labels=[1,0,0,1])
            m.drawmeridians(np.arange(self.lon_min,self.lon_max,self.d_lat),labels=[1,0,0,1])
        elif self.projection=='global':
            m=Basemap(projection='ortho',lon_0=self.lon_centre,lat_0=self.lat_centre,resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color=[1.0,1.0,1.0])
        # if colormap=='tomo':
        #     my_colormap=colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], \
        #         0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], \
        #         0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        # elif colormap=='mono':
        #     my_colormap=colormaps.make_colormap({0.0:[1.0,1.0,1.0], 0.15:[1.0,1.0,1.0], 0.85:[0.0,0.0,0.0], 1.0:[0.0,0.0,0.0]})
        try:
            geopolygons.PlotPolygon(mybasemap=m)
        except:
            pass
        #- loop over subvolumes
        for k in np.arange(self.nsubvol):
            if modelname =='dvsv':
                v = self.m[k].dvsv 
            if modelname =='dvsh':
                v = self.m[k].dvsh 
            if modelname =='drho':
                v = self.m[k].drho 
            if modelname =='dvp':
                v = self.m[k].dvp
            depth=np.zeros(np.shape(v[:,:,0]))
            nx=len(self.m[k].lat)
            ny=len(self.m[k].lon)
            #- find depth
            r=self.m[k].r
            r=0.5*(r[0:len(r)-1]+r[1:len(r)])
            for idx in np.arange(nx-1):
                for idy in np.arange(ny-1):
                    n=v[idx, idy,:]>=val
                    try:
                        depth[idx, idy]=6371.0-np.max(r[n])
                    except:
                        if len(r[n])==0:
                            depth[idx, idy]=6371.0-np.min(r)
                    # depth[idx, idy]=6371.0-r[n]
          #- rotate coordinate system if necessary
            lon,lat=np.meshgrid(self.m[k].lon[0:ny], self.m[k].lat[0:nx])
            if self.phi!=0.0:
                lat_rot=np.zeros(np.shape(lon),dtype=float)
                lon_rot=np.zeros(np.shape(lat),dtype=float)
                lat_rot, lon_rot = rotations.rotate_lat_lon(lat, lon, self.n, -self.phi) 
                # for idx in np.arange(nx):
                #     for idy in np.arange(ny):
                #         colat=90.0-lat[idx,idy]
                #         lat_rot[idx, idy], lon_rot[idx, idy]=rotate_coordinates(self.n,-self.phi, colat, lon[idx,idy])
                #         lat_rot[idx, idy]=90.0-lat_rot[idx, idy]
                lon=lon_rot
                lat=lat_rot
        #- convert to map coordinates and plot
            x,y=m(lon,lat)
            im=m.pcolor(x, y, depth, cmap=colormap, vmin=min_val_plot, vmax=max_val_plot, shading='gouraud')
        m.colorbar(im,"right", size="3%", pad='2%')
        plt.title('depth to '+str(val)+' km/s [km]')
        plt.show()
        return
