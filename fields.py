# -*- coding: utf-8 -*-
"""
A python module for SES3D binary field file manipulation.
Modified from python script in SES3D package( Andreas Fichtner and Lion Krischer)
    
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
import field2d_earth
import matplotlib.gridspec as gridspec
import h5py

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

class ses3d_fields(object):
    """
    Class for reading and plotting 3D fields defined on the SEM grid of SES3D.
    """
    def __init__(self, directory, setupfile, rotationfile=None, recfile='', field_type="earth_model"):
        """
        Initiate the ses3d_fields class.
        Read available components.
        ================================================================================================
        Input parameters:
        directory    - input data directory
        setupfile    - full path for setup file
        rotationfile - rotation parameter file(default is None)
        recfile      - receiver file
        field_type   - field type("earth_model", "velocity_snapshot", and "kernel")
        ================================================================================================
        """
        self.directory = directory
        self.field_type = field_type
        #- Read available Earth model files. ------------------------------------------------------
        if field_type == "earth_model":
            self.pure_components = ["A", "B", "C", "lambda", "mu", "rhoinv", "Q"]
            self.derived_components = ["vp", "vsh", "vsv", "rho"]
        #- Read available velocity snapshots
        if field_type == "velocity_snapshot":
            self.pure_components = ["vx", "vy", "vz"]
            self.derived_components = {}
        #- Read available kernels. ----------------------------------------------------------------
        if field_type == "kernel":
            self.pure_components = ["cp", "csh", "csv", "rho", "Q_mu", "Q_kappa", "alpha_mu", "alpha_kappa"]
            self.derived_components = {}
        #- Read setup file and make coordinates
        self.setup = self.read_setup(setupfile=setupfile)
        self.make_coordinates()
        #- Read rotation parameters. --------------------------------------------------------------
        if rotationfile == None:
            self.rotangle = 0.0
            self.n = np.array([0.0,1.0,0.0])
        else:
            with open(rotationfile,'r') as fid:
                fid.readline()
                self.rotangle = float(fid.readline().strip())
                fid.readline()
                line = fid.readline().strip().split(' ')
                self.n = np.array([float(line[0]),float(line[1]),float(line[2])])
        #- Read station locations, if available. --------------------------------------------------
        if os.path.exists(recfile):
            self.stations = True
            f = open(recfile)
            self.n_stations = int(f.readline())
            self.stnames = []
            self.stlats = []
            self.stlons = []
            for n in xrange(self.n_stations):
                self.stnames.append(f.readline().strip())
                dummy = f.readline().strip().split(' ')
                self.stlats.append(90.0-float(dummy[0]))
                self.stlons.append(float(dummy[1]))
            f.close()
        else:
            self.stations = False
        return;

    def read_setup(self, setupfile, verbose=True):
        """Read the setup file to get domain geometry.
        """
        setup = {}
        #- Open setup file and read header. -------------------------------------------------------
        if not os.path.isfile(setupfile): raise NameError('setup file does not exists!');
        with open(setupfile,'r') as f:
            lines = f.readlines()[1:]
            lines = [_i.strip() for _i in lines if _i.strip()]
            #- Read computational domain. -------------------------------------------------------------
            domain = {}
            domain["theta_min"] = float(lines.pop(0).split(' ')[0]) * np.pi / 180.0
            domain["theta_max"] = float(lines.pop(0).split(' ')[0]) * np.pi / 180.0
            domain["phi_min"]   = float(lines.pop(0).split(' ')[0]) * np.pi / 180.0
            domain["phi_max"]   = float(lines.pop(0).split(' ')[0]) * np.pi / 180.0
            domain["z_min"]     = float(lines.pop(0).split(' ')[0])
            domain["z_max"]     = float(lines.pop(0).split(' ')[0])
            setup["domain"]     = domain
            #- Read computational setup. --------------------------------------------------------------
            lines.pop(0)
            lines.pop(0)
            lines.pop(0)
            elements = {}
            elements["nx_global"] = int(lines.pop(0).split(' ')[0])
            elements["ny_global"] = int(lines.pop(0).split(' ')[0])
            elements["nz_global"] = int(lines.pop(0).split(' ')[0])
            setup["lpd"] = int(lines.pop(0).split(' ')[0])
            procs = {}
            procs["px"] = int(lines.pop(0).split(' ')[0])
            procs["py"] = int(lines.pop(0).split(' ')[0])
            procs["pz"] = int(lines.pop(0).split(' ')[0])
            setup["procs"] = procs
            elements["nx"] = 1 + elements["nx_global"] / procs["px"]
            elements["ny"] = 1 + elements["ny_global"] / procs["py"]
            elements["nz"] = 1 + elements["nz_global"] / procs["pz"]
            setup["elements"] = elements
        if verbose:
            print '====================== Reading setup parameters ======================'
            print 'domain:'
            for ikey in domain.keys(): print ikey,'=',str(domain[ikey])
            print 'elements:'
            for ikey in elements.keys(): print ikey,'=',str(elements[ikey])
            print 'lpd:',str(setup['lpd'])
            print '======================================================================'
        return setup
                                 
    def make_coordinates(self):
        """
        Make the coordinate lines for the different processor boxes.
        """
        n_procs     = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        #- Boundaries of the processor blocks. ----------------------------------------------------
        width_theta = (self.setup["domain"]["theta_max"] - self.setup["domain"]["theta_min"]) / self.setup["procs"]["px"]
        width_phi   = (self.setup["domain"]["phi_max"] - self.setup["domain"]["phi_min"]) / self.setup["procs"]["py"]
        width_z     = (self.setup["domain"]["z_max"] - self.setup["domain"]["z_min"]) / self.setup["procs"]["pz"]
        boundaries_theta    = np.arange(self.setup["domain"]["theta_min"],self.setup["domain"]["theta_max"]+width_theta,width_theta)
        boundaries_phi      = np.arange(self.setup["domain"]["phi_min"],self.setup["domain"]["phi_max"]+width_phi,width_phi)
        boundaries_z        = np.arange(self.setup["domain"]["z_min"],self.setup["domain"]["z_max"]+width_z,width_z)
        #- Make knot lines. -----------------------------------------------------------------------
        knot_x = self.get_GLL() + 1.0
        for ix in np.arange(self.setup["elements"]["nx"] - 1):
            knot_x = np.append(knot_x,self.get_GLL() + 1 + 2*(ix+1))
        knot_y = self.get_GLL() + 1.0
        for iy in np.arange(self.setup["elements"]["ny"] - 1):
            knot_y = np.append(knot_y,self.get_GLL() + 1 + 2*(iy+1))
        knot_z = self.get_GLL() + 1.0
        for iz in np.arange(self.setup["elements"]["nz"] - 1):
            knot_z = np.append(knot_z,self.get_GLL() + 1 + 2*(iz+1))
        knot_x = knot_x * width_theta / np.max(knot_x)
        knot_y = knot_y * width_phi / np.max(knot_y)
        knot_z = knot_z * width_z / np.max(knot_z)
        #- Loop over all processors. --------------------------------------------------------------
        self.theta  = np.empty(shape=(n_procs,len(knot_x)))
        self.phi    = np.empty(shape=(n_procs,len(knot_y)))
        self.z      = np.empty(shape=(n_procs,len(knot_z)))
        p = 0
        for iz in np.arange(self.setup["procs"]["pz"]):
            for iy in np.arange(self.setup["procs"]["py"]):
                for ix in np.arange(self.setup["procs"]["px"]):
                    self.theta[p,:] = boundaries_theta[ix] + knot_x
                    self.phi[p,:]   = boundaries_phi[iy] + knot_y
                    self.z[p,: :-1] = boundaries_z[iz] + knot_z
                    p += 1;
        return

    def get_GLL(self):
        """Set Gauss-Lobatto-Legendre(GLL) points for a given Lagrange polynomial degree.
        """
        if self.setup["lpd"] == 2:
            knots = np.array([-1.0, 0.0, 1.0])
        elif self.setup["lpd"] == 3:
            knots = np.array([-1.0, -0.4472135954999579, 0.4472135954999579, 1.0])
        elif self.setup["lpd"] == 4:
            knots = np.array([-1.0, -0.6546536707079772, 0.0, 0.6546536707079772, 1.0])
        elif self.setup["lpd"] == 5:
            knots = np.array([-1.0, -0.7650553239294647, -0.2852315164806451, 0.2852315164806451, 0.7650553239294647, 1.0])
        elif self.setup["lpd"] == 6:
            knots = np.array([-1.0, -0.8302238962785670, -0.4688487934707142, 0.0, 0.4688487934707142, 0.8302238962785670, 1.0])
        elif self.setup["lpd"] == 7:
            knots = np.array([-1.0, -0.8717401485096066, -0.5917001814331423,\
                -0.2092992179024789, 0.2092992179024789, 0.5917001814331423, 0.8717401485096066, 1.0])
        return knots

    def compose_filenames(self, component, proc_number, iteration=0):
        """Build filenames for the different field types.
        """
        # - Earth models. --------------------------------------------------------------------------
        if self.field_type == "earth_model":
            filename = os.path.join(self.directory, component+str(proc_number))
        # - Velocity field snapshots. --------------------------------------------------------------
        elif self.field_type == "velocity_snapshot":
            filename = os.path.join(self.directory, component+"_"+str(proc_number)+"_"+str(iteration))
        # - Sensitivity kernels. -------------------------------------------------------------------
        elif self.field_type == "kernel":
            filename = os.path.join(self.directory, "grad_"+component+"_"+str(proc_number))
        return filename;

    def read_single_box(self, component, proc_number, iteration=0):
        """Read the field from one single processor box.
        """
        # - Shape of the Fortran binary file. ------------------------------------------------------
        shape = (self.setup["elements"]["nx"],self.setup["elements"]["ny"],\
            self.setup["elements"]["nz"],self.setup["lpd"]+1,self.setup["lpd"]+1,self.setup["lpd"]+1)
        # - Read and compute the proper components. ------------------------------------------------
        if component in self.pure_components:
            filename = self.compose_filenames(component, proc_number, iteration)
            with open(filename, "rb") as open_file:
                field = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
        elif component in self.derived_components:
            # - rho 
            if component == "rho":
                filename = self.compose_filenames("rhoinv", proc_number, 0)
                with open(filename, "rb") as open_file:
                    field = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                field = 1.0 / field
            # - vp
            if component == "vp":
                filename1 = self.compose_filenames("lambda", proc_number, 0)
                filename2 = self.compose_filenames("mu", proc_number, 0)
                filename3 = self.compose_filenames("rhoinv", proc_number, 0)
                with open(filename1, "rb") as open_file:
                    field1 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                with open(filename2, "rb") as open_file:
                    field2 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                with open(filename3, "rb") as open_file:
                    field3 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                field = np.sqrt((field1 + 2 * field2) * field3)
            # - vsh
            if component == "vsh":
                filename1 = self.compose_filenames("mu", proc_number, 0)
                filename2 = self.compose_filenames("rhoinv", proc_number, 0)
                with open(filename1, "rb") as open_file:
                    field1 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                with open(filename2, "rb") as open_file:
                    field2 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                field = np.sqrt(field1 * field2)
            # - vsv
            if component == "vsv":
                filename1 = self.compose_filenames("mu", proc_number, 0)
                filename2 = self.compose_filenames("rhoinv", proc_number, 0)
                filename3 = self.compose_filenames("B", proc_number, 0)
                with open(filename1, "rb") as open_file:
                    field1 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                with open(filename2, "rb") as open_file:
                    field2 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                with open(filename3, "rb") as open_file:
                    field3 = np.ndarray(shape, buffer=open_file.read()[4:-4], dtype="float32", order="F")
                field = np.sqrt((field1 + field3) * field2)
        # - Reshape the array. ---------------------------------------------------------------------
        new_shape = [_i * _j for _i, _j in zip(shape[:3], shape[3:])]
        field = np.rollaxis(np.rollaxis(field, 3, 1), 3, self.setup["lpd"] + 1)
        field = field.reshape(new_shape, order="C")
        return field
    
    def check_model(self, component):
        """Check the minimum and maximum value of the model parameters
        """
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        vmax = float("-inf")
        vmin = float("inf")
        for p in range(n_procs):
            # - Read this field and make lats & lons. ------------------------------------------
            field = self.read_single_box(component,p,0)
            # - Find min and max values. -------------------------------------------------------
            vmax = max(vmax, field[:,:,:].max())
            vmin = min(vmin, field[:,:,:].min())
        print 'vmin=',vmin,'vmax=',vmax
        return
    
    
    def convert_to_hdf5(self, outfname, component, depth,  iter0=0, iterf=30000, diter=100, verbose=True ):
        """
        Plot depth slices of field component at given depth ranging between "valmin" and "valmax"
        ================================================================================================
        Input parameters:
        outfname        - output hdf5 file name
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        depth           - depth for plot (km)
        iter0, iterf    - start/end iteration index
        diter           - iteration interval
        =================================================================================================
        """
        if not( (component in self.pure_components) or (component in self.derived_components) ):
            raise TypeError('Incompatible component: '+component+' with field type: '+self.field_type)
        # - Some initialisations. ------------------------------------------------------------------
        dset    = h5py.File(outfname)
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        radius = 1000.0 * (6371.0 - depth)
        dset.attrs.create(name = 'theta_max', data=self.setup["domain"]["theta_max"], dtype='f')
        dset.attrs.create(name = 'theta_min', data=self.setup["domain"]["theta_min"], dtype='f')
        dset.attrs.create(name = 'phi_min', data=self.setup["domain"]["phi_min"], dtype='f')
        dset.attrs.create(name = 'phi_max', data=self.setup["domain"]["phi_max"], dtype='f')
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        dset.attrs.create(name = 'lat_min', data=lat_min, dtype='f')
        dset.attrs.create(name = 'lat_max', data=lat_max, dtype='f')
        dset.attrs.create(name = 'lon_min', data=lon_min, dtype='f')
        dset.attrs.create(name = 'lon_max', data=lon_max, dtype='f')
        dset.attrs.create(name = 'depth', data=depth, dtype='f')
        dset.attrs.create(name = 'n_procs', data=n_procs, dtype='f')
        dset.attrs.create(name = 'rotation_axis', data=self.n, dtype='f')
        dset.attrs.create(name = 'rotation_angle', data=self.rotangle, dtype='f')
        group   = dset.create_group( name = component ) 
        # - Loop over processor boxes and check if depth falls within the volume. ------------------
        iterArr=np.arange(iter0 ,iterf+diter, diter, dtype=int)
        for iteration in iterArr:
            if verbose: print 'Converting snapshot to hdf5 for iteration =',iteration
            try:
                for p in xrange(n_procs):
                    if (radius >= self.z[p,:].min()) & (radius <= self.z[p,:].max()):
                        # - Read this field and make lats & lons. ------------------------------------------
                        idz     = min(np.where(min(np.abs(self.z[p,:]-radius))==np.abs(self.z[p,:]-radius))[0])
                        field   = (self.read_single_box(component, p, iteration))[:,:,idz]
                        subgroup=group.require_group(name=str(iteration))
                        subdset = subgroup.create_dataset(name=str(p), shape=field.shape, data=field)
                        subdset.attrs.create(name = 'theta', data=self.theta[p,:], dtype='f')
                        subdset.attrs.create(name = 'phi', data=self.phi[p,:], dtype='f')
            except IOError: print 'iteration:',iteration,' NOT exists!'
        dset.close()
        return
    
    
    
    
    def plot_lat_slice(self, component, lat, valmin, valmax, iteration=0):
        """
        Plot slice at constant colatitude.
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        lat             - latitude for plot
        valmin, valmax  - minimum/maximum value for plotting
        iteration       - only required for snapshot plot
        =================================================================================================
        """
        # - Some initialisations. ------------------------------------------------------------------
        colat = np.pi * (90.-lat) / 180.0
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        vmax = float("-inf")
        vmin = float("inf")
        fig, ax = plt.subplots()
        # - Loop over processor boxes and check if colat falls within the volume. ------------------
        for p in range(n_procs):
            if (colat >= self.theta[p,:].min()) & (colat <= self.theta[p,:].max()):
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component,p,iteration)
                r, lon = np.meshgrid(self.z[p,:], self.phi[p,:])
                x = r * np.cos(lon)
                y = r * np.sin(lon)
                # - Find the colat index and plot for this one box. --------------------------------
                idx=min(np.where(min(np.abs(self.theta[p,:]-colat))==np.abs(self.theta[p,:]-colat))[0])
                colat_effective = self.theta[p,idx]*180.0/np.pi
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[idx,:,:].max())
                vmin = min(vmin, field[idx,:,:].min())
                cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
                cax = ax.pcolor(x, y, field[idx,:,:], cmap=cmap, vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = fig.colorbar(cax)
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        plt.suptitle("Vertical slice of %s at %i degree colatitude" % (component, colat_effective), size="large")
        plt.axis('equal')
        plt.show()
        return
    
    def plot_lat_depth_lon_slice(self, component, lat, depth, minlon, maxlon, valmin, valmax, iteration=0):
        """
        Plot slice at constant colatitude within given longitude range
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        lat             - latitude for plot
        minlon, maxlon  - minimum/maximum longitude
        valmin, valmax  - minimum/maximum value for plotting
        iteration       - only required for snapshot plot
        =================================================================================================
        """
        # - Some initialisations. ------------------------------------------------------------------
        colat = np.pi * (90.-lat) / 180.0
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        vmax = float("-inf")
        vmin = float("inf")
        radius = 1000.0 * (6371.0 - depth)
        minlon=np.pi*minlon/180.0
        maxlon=np.pi*maxlon/180.0
        fig, ax = plt.subplots()
        # - Loop over processor boxes and check if colat falls within the volume. ------------------
        for p in range(n_procs):
            if (colat >= self.theta[p,:].min()) and (colat <= self.theta[p,:].max()) and (radius <= self.z[p,:].max())\
                and (minlon <= self.phi[p,:].min()) and (maxlon >= self.phi[p,:].max()):
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component,p,iteration)
                r, lon = np.meshgrid(self.z[p,:], self.phi[p,:])
                x = r * np.cos(lon)
                y = r * np.sin(lon)
                # - Find the colat index and plot for this one box. --------------------------------
                idx=min(np.where(min(np.abs(self.theta[p,:]-colat))==np.abs(self.theta[p,:]-colat))[0])
                colat_effective = self.theta[p,idx]*180.0/np.pi
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[idx,:,:].max())
                vmin = min(vmin, field[idx,:,:].min())
                # - Make a nice colourmap and plot. ------------------------------------------------
                cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
                cax = ax.pcolor(x, y, field[idx,:,:], cmap=cmap, vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = fig.colorbar(cax)
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        plt.suptitle("Vertical slice of %s at %i degree colatitude" % (component, colat_effective), size="large")
        plt.axis('equal')
        plt.show()
        return

    def plot_depth_slice(self, component, depth, valmin, valmax, iteration=0, stations=True,
            res="l", proj='regional_ortho', zoomin=2, geopolygons=None, evlo=None, evla=None):
        """
        Plot depth slices of field component at given depth ranging between "valmin" and "valmax"
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        depth           - depth for plot (km)
        valmin, valmax  - minimum/maximum value for plotting
        iteration       - only required for snapshot plot
        stations        - plot stations or not
        res             - resolution of the coastline (c, l, i, h, f)
        proj            - projection type (global, regional_ortho, regional_merc)
        zoomin          - zoom in factor for proj = regional_ortho
        geopolygons     - geological polygons( basins etc. ) for plot
        =================================================================================================
        """
        if not( (component in self.pure_components) or (component in self.derived_components) ):
            raise TypeError('Incompatible component: '+component+' with field type: '+self.field_type)
        # - Some initialisations. ------------------------------------------------------------------
        fig=plt.figure()
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        radius = 1000.0 * (6371.0 - depth)
        vmax = float("-inf")
        vmin = float("inf")
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        lat_centre,lon_centre = rotations.rotate_lat_lon(lat_centre, lon_centre, self.n, -self.rotangle)
        # lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
        lat_centre = 90.0-lat_centre
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)
        # - Set up the map. ------------------------------------------------------------------------
        if proj=='global':
            m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif proj=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
            m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/zoomin, urcrnry=m1.urcrnry/3.5)
            # labels = [left,right,top,bottom]
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0))	
            # m.drawparallels(np.arange(-90.,120.,30.))
            # m.drawmeridians(np.arange(0.,360.,60.))
        elif proj=='regional_merc':
            m=Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                    llcrnrlon=lon_min, urcrnrlon=lon_max, lat_ts=20,resolution=res)
            m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
            m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        elif proj=='lambert':
            distEW, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                                lat_min, lon_max) # distance is in m
            distNS, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                                lat_max+1.7, lon_min) # distance is in m
            m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                lat_1=lat_min, lat_2=lat_max, lon_0=lon_centre, lat_0=lat_centre+1.2)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=2, dashes=[2,2], labels=[1,0,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=2, dashes=[2,2], labels=[0,0,1,1], fontsize=15)
        m.drawcoastlines()
        m.fillcontinents(lake_color='white',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        m.drawcountries()
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        try:
            evx, evy=m(evlo, evla)
            m.plot(evx, evy, 'yo', markersize=2)
        except:
            pass
        # - Loop over processor boxes and check if depth falls within the volume. ------------------
        for p in range(n_procs):
            if (radius >= self.z[p,:].min()) & (radius <= self.z[p,:].max()):
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component,p,iteration)
                # lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                # lons = self.phi[p,:] * 180.0 / np.pi
                # lon, lat = np.meshgrid(lons, lats)
                # - Find the depth index and plot for this one box. --------------------------------
                idz=min(np.where(min(np.abs(self.z[p,:]-radius))==np.abs(self.z[p,:]-radius))[0])
                r_effective = int(self.z[p,idz]/1000.0)
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[:,:,idz].max())
                vmin = min(vmin, field[:,:,idz].min())
                # - Make lats and lons. ------------------------------------------------------------
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Rotate if necessary. -----------------------------------------------------------
                if self.rotangle != 0.0:
                    lat_rot = np.zeros(np.shape(lon),dtype=float)
                    lon_rot = np.zeros(np.shape(lat),dtype=float)
                    for idlon in np.arange(len(lons)):
                        for idlat in np.arange(len(lats)):
                            # lat_rot[idlat,idlon],lon_rot[idlat,idlon] = rotate_coordinates(self.n,-self.rotangle,90.0-lat[idlat,idlon],lon[idlat,idlon])
                            lat_rot[idlat,idlon],lon_rot[idlat,idlon]  = rotations.rotate_lat_lon(lat[idlat,idlon], lon[idlat,idlon],  self.n, -self.rotangle)
                            lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                    lon = lon_rot
                    lat = lat_rot
                # - colourmap. ---------------------------------------------------------
                cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
                x, y = m(lon, lat)
                im = m.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap=cmap, vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = m.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        plt.suptitle("Depth slice of %s at %i km" % (component, r_effective), fontsize=20)
        # - Plot stations if available. ------------------------------------------------------------
        if self.stations and stations:
            x,y = m(self.stlons,self.stlats)
            for n in range(self.n_stations):
                plt.text(x[n],y[n],self.stnames[n][:4])
                plt.plot(x[n],y[n],'ro')
        plt.show()
        print "minimum value: "+str(vmin)+", maximum value: "+str(vmax)
        return
    
    def plot_depth_padding_slice(self, component, depth, valmin, valmax, dt, evla, evlo, vpadding=2.5, iteration=0, stations=True,
            res="l", proj='regional_ortho', zoomin=2, geopolygons=None):
        """
        Plot depth slices of field component at given depth ranging between "valmin" and "valmax"
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        depth           - depth for plot (km)
        valmin, valmax  - minimum/maximum value for plotting
        iteration       - only required for snapshot plot
        stations        - plot stations or not
        res             - resolution of the coastline (c, l, i, h, f)
        proj            - projection type (global, regional_ortho, regional_merc)
        zoomin          - zoom in factor for proj = regional_ortho
        geopolygons     - geological polygons( basins etc. ) for plot
        =================================================================================================
        """
        if not( (component in self.pure_components) or (component in self.derived_components) ):
            raise TypeError('Incompatible component: '+component+' with field type: '+self.field_type)
        # - Some initialisations. ------------------------------------------------------------------
        fig=plt.figure()
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        radius = 1000.0 * (6371.0 - depth)
        vmax = float("-inf")
        vmin = float("inf")
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        lat_centre,lon_centre = rotations.rotate_lat_lon(lat_centre, lon_centre, self.n, -self.rotangle)
        # lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
        # lat_centre = 90.0-lat_centre
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)
        # - Set up the map. ------------------------------------------------------------------------
        if proj=='global':
            m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif proj=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
            m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/zoomin, urcrnry=m1.urcrnry/3.5)
            # labels = [left,right,top,bottom]
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0))	
            # m.drawparallels(np.arange(-90.,120.,30.))
            # m.drawmeridians(np.arange(0.,360.,60.))
        elif proj=='regional_merc':
            m=Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                    llcrnrlon=lon_min, urcrnrlon=lon_max, lat_ts=20,resolution=res)
            m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
            m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        elif proj=='lambert':
            distEW, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                                lat_min, lon_max) # distance is in m
            distNS, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                                lat_max+2., lon_min) # distance is in m
            m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                lat_1=lat_min, lat_2=lat_max, lat_0=lat_centre+1.5, lon_0=lon_centre)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=2, dashes=[2,2], labels=[1,0,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=2, dashes=[2,2], labels=[0,0,1,1], fontsize=15)
        m.drawcoastlines()
        m.fillcontinents(lake_color='white',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        m.drawcountries()
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        mindist = dt * iteration * vpadding
        evx, evy=m(evlo, evla)
        m.plot(evx, evy, 'yo', markersize=2)

        # - Loop over processor boxes and check if depth falls within the volume. ------------------
        for p in range(n_procs):
            latArr = 90. - self.theta[p,:]*180.0/np.pi
            lonArr = self.phi[p,:]*180.0/np.pi
            pmaxdist = max( great_circle(( latArr.min(), lonArr.min() ), (evla, evlo)).km, great_circle(( latArr.max(), lonArr.min() ), (evla, evlo)).km,
                    great_circle(( latArr.min(), lonArr.max() ), (evla, evlo)).km, great_circle(( latArr.max(), lonArr.max() ), (evla, evlo)).km)
            if (radius >= self.z[p,:].min()) and (radius <= self.z[p,:].max()):
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component, p, iteration)
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Find the depth index and plot for this one box. --------------------------------
                idz=min(np.where(min(np.abs(self.z[p,:]-radius))==np.abs(self.z[p,:]-radius))[0])
                r_effective = int(self.z[p,idz]/1000.0)
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[:,:,idz].max())
                vmin = min(vmin, field[:,:,idz].min())
                # - Make lats and lons. ------------------------------------------------------------
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Rotate if necessary. -----------------------------------------------------------
                if self.rotangle != 0.0:
                    lat_rot = np.zeros(np.shape(lon),dtype=float)
                    lon_rot = np.zeros(np.shape(lat),dtype=float)
                    for idlon in np.arange(len(lons)):
                        for idlat in np.arange(len(lats)):
                            # lat_rot[idlat,idlon],lon_rot[idlat,idlon] = rotate_coordinates(self.n,-self.rotangle,90.0-lat[idlat,idlon],lon[idlat,idlon])
                            lat_rot[idlat,idlon],lon_rot[idlat,idlon]  = rotations.rotate_lat_lon(lat[idlat,idlon], lon[idlat,idlon],  self.n, -self.rotangle)
                            lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                    lon = lon_rot
                    lat = lat_rot
                # - colourmap. ---------------------------------------------------------
                cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
                x, y = m(lon, lat)
                if pmaxdist > mindist:
                    im = m.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap=cmap, vmin=valmin,vmax=valmax)
                else:
                    im = m.pcolormesh(x, y, np.zeros(lon.shape), shading='gouraud', cmap=cmap, vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = m.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        plt.suptitle("Depth slice of %s at %i km" % (component, r_effective), fontsize=20)
        # - Plot stations if available. ------------------------------------------------------------
        if self.stations and stations:
            x,y = m(self.stlons,self.stlats)
            for n in range(self.n_stations):
                plt.text(x[n],y[n],self.stnames[n][:4])
                plt.plot(x[n],y[n],'ro')
        plt.show()
        print "minimum value: "+str(vmin)+", maximum value: "+str(vmax)
        return
    
    
    def plot_depth_padding_all6_slice(self, component, depth, valmin, valmax, dt, evla, evlo, vpadding=2.5, iteration=0, stations=True,
            res="l", proj='regional_ortho', zoomin=2, geopolygons=None):
        """
        Plot depth slices of field component at given depth ranging between "valmin" and "valmax"
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        depth           - depth for plot (km)
        valmin, valmax  - minimum/maximum value for plotting
        iteration       - only required for snapshot plot
        stations        - plot stations or not
        res             - resolution of the coastline (c, l, i, h, f)
        proj            - projection type (global, regional_ortho, regional_merc)
        zoomin          - zoom in factor for proj = regional_ortho
        geopolygons     - geological polygons( basins etc. ) for plot
        =================================================================================================
        """
        minlat=23.
        maxlat=51.
        minlon=86.
        maxlon=132.
        Tfield=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)
        Tfield.read_dbase(datadir='./output_ses3d_all6')
        Afield=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10., fieldtype='Amp')
        Afield.cut_edge(1,1)
        Afield.read_dbase(datadir='./output_ses3d_all6')
        # field.get_distArr()
        if not( (component in self.pure_components) or (component in self.derived_components) ):
            raise TypeError('Incompatible component: '+component+' with field type: '+self.field_type)
        # - Some initialisations. ------------------------------------------------------------------
        fig = plt.figure()
        ax = fig.add_subplot(231)
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        radius = 1000.0 * (6371.0 - depth)
        vmax = float("-inf")
        vmin = float("inf")
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        lat_centre,lon_centre = rotations.rotate_lat_lon(lat_centre, lon_centre, self.n, -self.rotangle)
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)
        # - Set up the map. ------------------------------------------------------------------------
        if proj=='global':
            m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif proj=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
            m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/zoomin, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0))	
        elif proj=='regional_merc':
            m=Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                    llcrnrlon=lon_min, urcrnrlon=lon_max, lat_ts=20,resolution=res)
            m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
            m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        elif proj=='lambert':
            distEW, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                                lat_min, lon_max) # distance is in m
            distNS, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                                lat_max+2., lon_min) # distance is in m
            m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                lat_1=lat_min, lat_2=lat_max, lat_0=lat_centre+1.5, lon_0=lon_centre)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=2, dashes=[2,2], labels=[1,0,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=2, dashes=[2,2], labels=[0,0,1,1], fontsize=15)
        m.drawcoastlines()
        m.fillcontinents(lake_color='white',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        m.drawcountries()
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        mindist = dt * iteration * vpadding
        evx, evy=m(evlo, evla)
        m.plot(evx, evy, 'yo', markersize=2)
        # - Loop over processor boxes and check if depth falls within the volume. ------------------
        for p in range(n_procs):
            latArr = 90. - self.theta[p,:]*180.0/np.pi
            lonArr = self.phi[p,:]*180.0/np.pi
            pmaxdist = max( great_circle(( latArr.min(), lonArr.min() ), (evla, evlo)).km, great_circle(( latArr.max(), lonArr.min() ), (evla, evlo)).km,
                    great_circle(( latArr.min(), lonArr.max() ), (evla, evlo)).km, great_circle(( latArr.max(), lonArr.max() ), (evla, evlo)).km)
            if (radius >= self.z[p,:].min()) and (radius <= self.z[p,:].max()):
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component, p, iteration)
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Find the depth index and plot for this one box. --------------------------------
                idz=min(np.where(min(np.abs(self.z[p,:]-radius))==np.abs(self.z[p,:]-radius))[0])
                r_effective = int(self.z[p,idz]/1000.0)
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[:,:,idz].max())
                vmin = min(vmin, field[:,:,idz].min())
                # - Make lats and lons. ------------------------------------------------------------
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Rotate if necessary. -----------------------------------------------------------
                if self.rotangle != 0.0:
                    lat_rot = np.zeros(np.shape(lon),dtype=float)
                    lon_rot = np.zeros(np.shape(lat),dtype=float)
                    for idlon in np.arange(len(lons)):
                        for idlat in np.arange(len(lats)):
                            # lat_rot[idlat,idlon],lon_rot[idlat,idlon] = rotate_coordinates(self.n,-self.rotangle,90.0-lat[idlat,idlon],lon[idlat,idlon])
                            lat_rot[idlat,idlon],lon_rot[idlat,idlon]  = rotations.rotate_lat_lon(lat[idlat,idlon], lon[idlat,idlon],  self.n, -self.rotangle)
                            lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                    lon = lon_rot
                    lat = lat_rot
                # - colourmap. ---------------------------------------------------------
                cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
                x, y = m(lon, lat)
                if pmaxdist > mindist:
                    im = m.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap=cmap, vmin=valmin,vmax=valmax)
                else:
                    im = m.pcolormesh(x, y, np.zeros(lon.shape), shading='gouraud', cmap=cmap, vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = m.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        # plt.suptitle("Depth slice of %s at %i km" % (component, r_effective), fontsize=20)
        # - Plot stations if available. ------------------------------------------------------------
        if self.stations and stations:
            x,y = m(self.stlons,self.stlats)
            for n in range(self.n_stations):
                plt.text(x[n],y[n],self.stnames[n][:4])
                plt.plot(x[n],y[n],'ro')
        ################################################################################
        maxdist=iteration*0.05*3.
        Tfield.reset_reason(dist=maxdist)
        Afield.reset_reason(dist=maxdist)
        Afield.np2ma()
        Tfield.np2ma()
        ax = fig.add_subplot(232)
        Tfield.plot_field(contour=True, showfig=False, vmin=0, vmax=1500.)
        ax = fig.add_subplot(233)
        Tfield.plot_diffa(showfig=False)
        ax = fig.add_subplot(234)
        Tfield.plot_appV(showfig=False)
        ax = fig.add_subplot(235)
        Afield.plot_field(contour=False,showfig=False, vmin=0, vmax=1200.)
        ax = fig.add_subplot(236)
        Afield.plot_lplcC(showfig=False, infield=Tfield)
        plt.suptitle('t = '+str(iteration*0.05)+' sec', fontsize=10, y=0.88)
        plt.show()
        print "minimum value: "+str(vmin)+", maximum value: "+str(vmax)
        return
    
    
    
    def plot_snapshots(self, component, depth, valmin, valmax, outdir, fprx='wavefield',iter0=100, iterf=17100, \
            diter=200, stations=False, res="i", proj='regional', dpi=300, zoomin=2, geopolygons=None, evlo=None, evla=None ):
        """
        Plot snapshots of field component at given depth ranging between "valmin" and "valmax"
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        depth           - depth for plot (km)
        valmin, valmax  - minimum/maximum value for plotting
        outdir          - output directory
        fprx            - output file name prefix
        iter0, iterf    - inital/final iterations for plotting
        diter           - iteration interval
        stations        - plot stations or not
        res             - resolution of the coastline (c, l, i, h, f)
        proj            - projection type (global, regional_ortho, regional_merc)
        dpi             - dots per inch (figure resolution parameter)
        zoomin          - zoom in factor for proj = regional_ortho
        geopolygons     - geological polygons( basins etc. ) for plot
        evlo, evla      - event location for plotting
        =================================================================================================
        """
        outdir=outdir+'_'+str(depth)+'km'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for iteration in np.arange((iterf-iter0)/diter)*diter+iter0:
            vfname=self.directory+'/vx_0_'+str(int(iteration));
            if not os.path.isfile(vfname):
                raise NameError('Velocity Snapshot:'+vfname+' does not exist!')
        fig=plt.figure(num=None, figsize=(10, 10), dpi=dpi, facecolor='w', edgecolor='k')
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        # lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
        lat_centre,lon_centre = rotations.rotate_lat_lon(lat_centre, lon_centre, self.n, -self.rotangle)
        lat_centre = 90.0-lat_centre
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)
        if proj=='global':
            m=Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
        elif proj=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
            m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/zoomin, urcrnry=m1.urcrnry/zoomin)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
        elif proj=='regional_merc':
            m=Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                        llcrnrlon=lon_min, urcrnrlon=lon_max, lat_ts=20, resolution=res)
            m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
            m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        elif proj=='lambert':
            distEW, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                                lat_min, lon_max) # distance is in m
            distNS, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                                lat_max+1.7, lon_min) # distance is in m
            m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                lat_1=lat_min, lat_2=lat_max, lon_0=lon_centre+1.2, lat_0=lat_centre,)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=2, dashes=[2,2], labels=[1,0,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=2, dashes=[2,2], labels=[0,0,1,1], fontsize=15)
        m.drawcoastlines()
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        m.drawcountries()
        try:
            evx, evy=m(evlo, evla)
            m.plot(evx, evy, 'yo', markersize=2)
        except:
            pass
        try:
            geopolygons.PlotPolygon(inbasemap=mymap)
        except:
            pass
        iterLst=np.arange((iterf-iter0)/diter)*diter+iter0;
        iterLst=iterLst.tolist();
        for iteration in np.arange((iterf-iter0)/diter)*diter+iter0:
            self._plot_snapshot(inmap=m, component=component, depth=depth, valmin=valmin, valmax=valmax, iteration=iteration, stations=stations)
            outfname=outdir+'/'+fprx+'_%06d.png' %(iteration)
            print outfname, outdir
            fig.savefig(outfname, format='png', dpi=dpi)
        return 
    
    def _plot_snapshot(self, inmap, component, depth, valmin, valmax, iteration, stations):
        """Plot snapshot, private function used by make_animation
        """
        print 'Plotting Snapshot for:',iteration,' steps!'
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        radius = 1000.0 * (6371.0 - depth)
        vmax = float("-inf")
        vmin = float("inf")
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        # lat_centre,lon_centre = rotate_coordinates(self.n,-self.rotangle,90.0-lat_centre,lon_centre)
        lat_centre,lon_centre = rotations.rotate_lat_lon(lat_centre, lon_centre, self.n, -self.rotangle)
        lat_centre = 90.0-lat_centre
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)
        # - Loop over processor boxes and check if depth falls within the volume. ------------------
        for p in range(n_procs):
            if (radius >= self.z[p,:].min()) & (radius <= self.z[p,:].max()):
                # - Read this field and make lats & lons. ------------------------------------------
                print p
                field = self.read_single_box(component,p,iteration)
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Find the depth index and plot for this one box. --------------------------------
                idz=min(np.where(min(np.abs(self.z[p,:]-radius))==np.abs(self.z[p,:]-radius))[0])
                r_effective = int(self.z[p,idz]/1000.0)
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[:,:,idz].max())
                vmin = min(vmin, field[:,:,idz].min())
                # - Make lats and lons. ------------------------------------------------------------
                lats = 90.0 - self.theta[p,:] * 180.0 / np.pi
                lons = self.phi[p,:] * 180.0 / np.pi
                lon, lat = np.meshgrid(lons, lats)
                # - Rotate if necessary. -----------------------------------------------------------
                if self.rotangle != 0.0:
                    lat_rot = np.zeros(np.shape(lon),dtype=float)
                    lon_rot = np.zeros(np.shape(lat),dtype=float)
                    for idlon in np.arange(len(lons)):
                        for idlat in np.arange(len(lats)):
                            lat_rot[idlat,idlon],lon_rot[idlat,idlon] = rotations.rotate_lat_lon(lat[idlat,idlon], lon[idlat,idlon], self.n, -self.rotangle)
                            lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                    lon = lon_rot
                    lat = lat_rot
                cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
                x, y = inmap(lon, lat)
                im=inmap.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap=cmap, vmin=valmin,vmax=valmax)
        # - Add colobar and title. ------------------------------------------------------------------
        cb = inmap.colorbar(im, "right", size="3%", pad='2%')
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        # - Plot stations if available. ------------------------------------------------------------
        if (self.stations == True) & (stations==True):
            x,y = mymap(self.stlons,self.stlats)
            for n in range(self.n_stations):
                plt.text(x[n],y[n],self.stnames[n][:4])
                plt.plot(x[n],y[n],'ro')
        return
    
    def plot_snapshots_mp(self, component, depth, valmin, valmax, outdir, fprx='wavefield',iter0=100, iterf=17100, diter=200,
            stations=False, res="i", proj='regional_ortho', dpi=300, zoomin=2, geopolygons=None, evlo=None, evla=None, vpadding=None, dt=None ):
        """Multiprocessing version of plot_snapshots
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        depth           - depth for plot (km)
        valmin, valmax  - minimum/maximum value for plotting
        outdir          - output directory
        fprx            - output file name prefix
        iter0, iterf    - inital/final iterations for plotting
        diter           - iteration interval
        stations        - plot stations or not
        res             - resolution of the coastline (c, l, i, h, f)
        proj            - projection type (global, regional_ortho, regional_merc)
        dpi             - dots per inch (figure resolution parameter)
        zoomin          - zoom in factor for proj = regional_ortho
        geopolygons     - geological polygons( basins etc. ) for plot
        evlo, evla      - event location for plotting 
        vpadding        - velocity for padding, if assigned, wavefield that is with dist = vpadding*time
                            will be padded to zero
        =================================================================================================
        """
        outdir=outdir+'_'+str(depth)+'km'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for iteration in np.arange((iterf-iter0)/diter)*diter+iter0:
            vfname=self.directory+'/vx_0_'+str(int(iteration))
            if not os.path.isfile(vfname):
                raise NameError('Velocity Snapshot:'+vfname+' does not exist!')
        # fig=plt.figure(num=None, figsize=(10, 10), dpi=dpi, facecolor='w', edgecolor='k')
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        lat_centre, lon_centre = rotations.rotate_lat_lon(lat_centre,lon_centre, self.n, -self.rotangle)
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)  
        iterLst=np.arange((iterf-iter0)/diter)*diter+iter0;
        iterLst=iterLst.tolist()
        PLOTSNAP = partial(Iter2snapshot, sfield=self, evlo=evlo, evla=evla, component=component, depth=depth,\
            valmin=valmin, valmax=valmax, stations=stations, fprx=fprx, proj=proj, \
            outdir=outdir, dpi=dpi, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,\
            lat_centre=lat_centre, lon_centre=lon_centre, d_lon=d_lon, d_lat=d_lat, res=res, \
            zoomin=zoomin, geopolygons=geopolygons, vpadding=vpadding, dt=dt)
        # PLOTSNAP(iterLst)
        pool=multiprocessing.Pool()
        pool.map(PLOTSNAP, iterLst) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Making Snapshots for Animation  ( MP ) !'
        return
    
    def plot_snapshots_all6_mp(self, component, depth, valmin, valmax, outdir, fprx='wavefield',iter0=100, iterf=17100, diter=200,
            stations=False, res="i", proj='regional_ortho', dpi=300, zoomin=2, geopolygons=None, evlo=None, evla=None, vpadding=None, dt=None ):
        """Multiprocessing version of plot_snapshots
        ================================================================================================
        Input parameters:
        component       - component for plotting
                            The currently available "components" are:
                                Material parameters: A, B, C, mu, lambda, rhoinv, vp, vsh, vsv, rho
                                Velocity field snapshots: vx, vy, vz
                                Sensitivity kernels: Q_mu, Q_kappa, alpha_mu, alpha_kappa
        depth           - depth for plot (km)
        valmin, valmax  - minimum/maximum value for plotting
        outdir          - output directory
        fprx            - output file name prefix
        iter0, iterf    - inital/final iterations for plotting
        diter           - iteration interval
        stations        - plot stations or not
        res             - resolution of the coastline (c, l, i, h, f)
        proj            - projection type (global, regional_ortho, regional_merc)
        dpi             - dots per inch (figure resolution parameter)
        zoomin          - zoom in factor for proj = regional_ortho
        geopolygons     - geological polygons( basins etc. ) for plot
        evlo, evla      - event location for plotting 
        vpadding        - velocity for padding, if assigned, wavefield that is with dist = vpadding*time
                            will be padded to zero
        =================================================================================================
        """
        outdir=outdir+'_'+str(depth)+'km'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for iteration in np.arange((iterf-iter0)/diter)*diter+iter0:
            vfname=self.directory+'/vx_0_'+str(int(iteration))
            if not os.path.isfile(vfname):
                raise NameError('Velocity Snapshot:'+vfname+' does not exist!')
        # fig=plt.figure(num=None, figsize=(10, 10), dpi=dpi, facecolor='w', edgecolor='k')
        lat_min = 90.0 - self.setup["domain"]["theta_max"]*180.0/np.pi
        lat_max = 90.0 - self.setup["domain"]["theta_min"]*180.0/np.pi
        lon_min = self.setup["domain"]["phi_min"]*180.0/np.pi
        lon_max = self.setup["domain"]["phi_max"]*180.0/np.pi
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        lat_centre, lon_centre = rotations.rotate_lat_lon(lat_centre,lon_centre, self.n, -self.rotangle)
        d_lon = np.round((lon_max-lon_min)/10.0)
        d_lat = np.round((lat_max-lat_min)/10.0)  
        iterLst=np.arange((iterf-iter0)/diter)*diter+iter0;
        iterLst=iterLst.tolist()
        PLOTSNAP = partial(Iter2snapshot_all6, sfield=self, evlo=evlo, evla=evla, component=component, depth=depth,\
            valmin=valmin, valmax=valmax, stations=stations, fprx=fprx, proj=proj, \
            outdir=outdir, dpi=dpi, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,\
            lat_centre=lat_centre, lon_centre=lon_centre, d_lon=d_lon, d_lat=d_lat, res=res, \
            zoomin=zoomin, geopolygons=geopolygons, vpadding=vpadding, dt=dt)
        pool=multiprocessing.Pool()
        pool.map(PLOTSNAP, iterLst) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Making Snapshots for Animation  ( MP ) !'
        return
    
    
    def MakeAnimationVCrossSection(self, component, lat, depth, minlon, maxlon, valmin, valmax, outdir, prefix='mywavefield_VCS',\
            iter0=100, iterf=17100, dsnap=100, dpi=300):
        
        outdir=outdir+'_VCS'+str(lat);
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
            vfname=self.directory+'/vx_0_'+str(int(iteration));
            if not os.path.isfile(vfname):
                raise NameError('Velocity Snapshot:'+vfname+' does not exist!')
        
        for iteration in np.arange((iterf-iter0)/dsnap)*dsnap+iter0:
            self.PlotSnapshot(mymap=mymap, component=component, depth=depth, valmin=valmin, valmax=valmax, iteration=iteration, stations=stations);
            outpsfname=outdir+'/'+prefix+'_%06d.png' %(iteration);
            fig.savefig(outpsfname, format='png', dpi=dpi)
        
    
    def PlotSnapshot_VCrossSection(self, component, lat, depth, minlon, maxlon, valmin, valmax, outfname=None, iteration=0, verbose=True):
        # - Some initialisations. ------------------------------------------------------------------
        fig=plt.figure(num=None, figsize=(10, 10), dpi=dpi, facecolor='w', edgecolor='k')
        colat = np.pi * (90.-lat) / 180.0
        n_procs = self.setup["procs"]["px"] * self.setup["procs"]["py"] * self.setup["procs"]["pz"]
        vmax = float("-inf")
        vmin = float("inf")
        radius = 1000.0 * (6371.0 - depth);
        minlon=np.pi*minlon/180.0;
        maxlon=np.pi*maxlon/180.0;
        fig, ax = plt.subplots()
        # - Loop over processor boxes and check if colat falls within the volume. ------------------
        for p in range(n_procs):
            if (colat >= self.theta[p,:].min()) and (colat <= self.theta[p,:].max()) and (radius <= self.z[p,:].max())\
                and (minlon <= self.phi[p,:].min()) and (maxlon >= self.phi[p,:].max()):
                print 6371.-self.z[p,:].min()/1000.
                # - Read this field and make lats & lons. ------------------------------------------
                field = self.read_single_box(component,p,iteration)
                r, lon = np.meshgrid(self.z[p,:], self.phi[p,:])
                x = r * np.cos(lon)
                y = r * np.sin(lon)
                # - Find the colat index and plot for this one box. --------------------------------
                idx=min(np.where(min(np.abs(self.theta[p,:]-colat))==np.abs(self.theta[p,:]-colat))[0])
                colat_effective = self.theta[p,idx]*180.0/np.pi
                # - Find min and max values. -------------------------------------------------------
                vmax = max(vmax, field[idx,:,:].max())
                vmin = min(vmin, field[idx,:,:].min())
                # - Make a nice colourmap and plot. ------------------------------------------------
                # my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],\
                #     0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
                cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
                cax = ax.pcolor(x, y, field[idx,:,:], cmap=cmap, vmin=valmin,vmax=valmax)
                # if outfname !=None:
                    
        # - Add colobar and title. ------------------------------------------------------------------
        cb = fig.colorbar(cax)
        if component in UNIT_DICT:
            cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
        plt.suptitle("Vertical slice of %s at %i degree colatitude" % (component, colat_effective), size="large")
        plt.axis('equal')
        plt.show()
        return
    
def Iter2snapshot(iterN, sfield, evlo, evla, component, depth, valmin, valmax, stations, fprx, proj, outdir, dpi, \
        lat_min, lat_max, lon_min, lon_max, lat_centre, lon_centre, d_lon, d_lat, res, zoomin, geopolygons, vpadding, dt):
    """Plot snapshot, used by plot_snapshots_mp
    """
    print 'Plotting Snapshot for:',iterN,' step!'
    n_procs = sfield.setup["procs"]["px"] * sfield.setup["procs"]["py"] * sfield.setup["procs"]["pz"]
    radius = 1000.0 * (6371.0 - depth);
    vmax = float("-inf");
    vmin = float("inf");
    # fig=plt.figure(num=iterN, figsize=(10, 10), dpi=dpi, facecolor='w', edgecolor='k')
    # - Set up the map. ------------------------------------------------------------------------
    if proj=='global':
        m=Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
        m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
        m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
    elif proj=='regional_ortho':
        m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
        m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
            llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/zoomin, urcrnry=m1.urcrnry/3.5)
    elif proj=='regional_merc':
        m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
        m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
        m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
    elif proj=='lambert':
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
    except:
        pass
    try:
        mindist = dt * iterN * vpadding
        print 'will do padding for iteration: %d' %iterN
    except:
        mindist = -1.
        print 'No padding for iteration: %d' %iterN
    # - Loop over processor boxes and check if depth falls within the volume. ------------------
    for p in range(n_procs):
        latArr = 90. - sfield.theta[p,:]*180.0/np.pi
        lonArr = sfield.phi[p,:]*180.0/np.pi
        pmaxdist = max( great_circle(( latArr.min(), lonArr.min() ), (evla, evlo)).km, great_circle(( latArr.max(), lonArr.min() ), (evla, evlo)).km,
                    great_circle(( latArr.min(), lonArr.max() ), (evla, evlo)).km, great_circle(( latArr.max(), lonArr.max() ), (evla, evlo)).km)
        if (radius >= sfield.z[p,:].min()) & (radius <= sfield.z[p,:].max()):
            # - Read this field and make lats & lons. ------------------------------------------
            field = sfield.read_single_box(component,p,iterN)
            lats = 90.0 - sfield.theta[p,:] * 180.0 / np.pi
            lons = sfield.phi[p,:] * 180.0 / np.pi
            lon, lat = np.meshgrid(lons, lats)
            # - Find the depth index and plot for this one box. --------------------------------
            idz=min(np.where(min(np.abs(sfield.z[p,:]-radius))==np.abs(sfield.z[p,:]-radius))[0])
            r_effective = int(sfield.z[p,idz]/1000.0)
            # - Find min and max values. -------------------------------------------------------
            vmax = max(vmax, field[:,:,idz].max())
            vmin = min(vmin, field[:,:,idz].min())
            # - Make lats and lons. ------------------------------------------------------------
            lats = 90.0 - sfield.theta[p,:] * 180.0 / np.pi
            lons = sfield.phi[p,:] * 180.0 / np.pi
            lon, lat = np.meshgrid(lons, lats)
            # - Rotate if necessary. -----------------------------------------------------------
            if sfield.rotangle != 0.0:
                lat_rot = np.zeros(np.shape(lon),dtype=float)
                lon_rot = np.zeros(np.shape(lat),dtype=float)
                for idlon in np.arange(len(lons)):
                    for idlat in np.arange(len(lats)):
                        # lat_rot[idlat,idlon],lon_rot[idlat,idlon] = rotate_coordinates(sfield.n,-sfield.rotangle,90.0-lat[idlat,idlon],lon[idlat,idlon])
                        lat_rot[idlat,idlon],lon_rot[idlat,idlon]  = rotations.rotate_lat_lon(lat[idlat,idlon],lon[idlat,idlon], sfield.n,-sfield.rotangle)
                        lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                lon = lon_rot
                lat = lat_rot
            # - Make a nice colourmap. ---------------------------------------------------------
            # my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],\
            #     0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
            # my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.3,0.0],\
            #     0.39:[1.0,0.7,0.0], 0.5:[0.92,0.92,0.92], 0.61:[0.0,0.6,0.7], 0.7:[0.0,0.3,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            # my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.3,0.0],\
            #     0.35:[1.0,0.7,0.0], 0.5:[0.92,0.92,0.92], 0.65:[0.0,0.6,0.7], 0.7:[0.0,0.3,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            x, y = m(lon, lat)
            if pmaxdist > mindist:
                im = m.pcolormesh(x, y, field[:,:,idz], shading='gouraud', cmap=cmap, vmin=valmin,vmax=valmax)
            else:
                im = m.pcolormesh(x, y, np.zeros(lon.shape), shading='gouraud', cmap=cmap, vmin=valmin,vmax=valmax)
    # - Add colobar and title. ------------------------------------------------------------------
    cb = m.colorbar(im, "right", size="3%", pad='2%')
    try:
        geopolygons.PlotPolygon(inbasemap=m)
    except:
        pass
    if component in UNIT_DICT:
        cb.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)
    # - Plot stations if available. ------------------------------------------------------------
    if (sfield.stations == True) & (stations==True):
        x,y = m(sfield.stlons,sfield.stlats)
        for n in range(sfield.n_stations):
            plt.text(x[n],y[n],sfield.stnames[n][:4])
            plt.plot(x[n],y[n],'ro')
    outfname=outdir+'/'+fprx+'_%06d.png' %(iterN)
    savefig(outfname, format='png', dpi=dpi)
    return

def Iter2snapshot_all6(iterN, sfield, evlo, evla, component, depth, valmin, valmax, stations, fprx, proj, outdir, dpi, \
        lat_min, lat_max, lon_min, lon_max, lat_centre, lon_centre, d_lon, d_lat, res, zoomin, geopolygons, vpadding, dt):
    """Plot snapshot, used by plot_snapshots_all6_mp
    """
    minlat=23.
    maxlat=51.
    minlon=86.
    maxlon=132.
    Tfield=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)
    Tfield.read_dbase(datadir='./output_ses3d_all6')
    Afield=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10., fieldtype='Amp')
    Afield.cut_edge(1,1)
    Afield.read_dbase(datadir='./output_ses3d_all6')
    print 'Plotting Snapshot for:',iterN,' step!'
    n_procs = sfield.setup["procs"]["px"] * sfield.setup["procs"]["py"] * sfield.setup["procs"]["pz"]
    radius = 1000.0 * (6371.0 - depth);
    vmax = float("-inf");
    vmin = float("inf");
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(2, 3)
    ax = plt.subplot(gs[0, 0])
    # - Set up the map. ------------------------------------------------------------------------
    if proj=='global':
        m=Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
        m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
        m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])	
    elif proj=='regional_ortho':
        m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
        m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
            llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/zoomin, urcrnry=m1.urcrnry/3.5)
    elif proj=='regional_merc':
        m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
        m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
        m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
    elif proj=='lambert':
        distEW, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                            lat_min, lon_max) # distance is in m
        distNS, az, baz=obspy.geodetics.gps2dist_azimuth(lat_min, lon_min,
                            lat_max+2., lon_min) # distance is in m
        m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
            lat_1=lat_min, lat_2=lat_max, lon_0=lon_centre, lat_0=lat_centre+1.5)
        m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=0.5, dashes=[2,2], labels=[0,0,0,0], fontsize=5)
        m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=0.5, dashes=[2,2], labels=[0,0,0,0], fontsize=5)
    m.drawcoastlines()
    m.fillcontinents(lake_color='#99ffff',zorder=0.2)
    m.drawmapboundary(fill_color="white")
    m.drawcountries()
    try:
        evx, evy=m(evlo, evla)
        m.plot(evx, evy, 'yo', markersize=2)
    except:
        pass
    try:
        mindist = dt * iterN * vpadding
        print 'will do padding for iteration: %d' %iterN
    except:
        mindist = -1.
        print 'No padding for iteration: %d' %iterN
    # - Loop over processor boxes and check if depth falls within the volume. ------------------
    for p in range(n_procs):
        latArr = 90. - sfield.theta[p,:]*180.0/np.pi
        lonArr = sfield.phi[p,:]*180.0/np.pi
        pmaxdist = max( great_circle(( latArr.min(), lonArr.min() ), (evla, evlo)).km, great_circle(( latArr.max(), lonArr.min() ), (evla, evlo)).km,
                    great_circle(( latArr.min(), lonArr.max() ), (evla, evlo)).km, great_circle(( latArr.max(), lonArr.max() ), (evla, evlo)).km)
        if (radius >= sfield.z[p,:].min()) & (radius <= sfield.z[p,:].max()):
            # - Read this field and make lats & lons. ------------------------------------------
            field = sfield.read_single_box(component,p,iterN)
            lats = 90.0 - sfield.theta[p,:] * 180.0 / np.pi
            lons = sfield.phi[p,:] * 180.0 / np.pi
            lon, lat = np.meshgrid(lons, lats)
            # - Find the depth index and plot for this one box. --------------------------------
            idz=min(np.where(min(np.abs(sfield.z[p,:]-radius))==np.abs(sfield.z[p,:]-radius))[0])
            r_effective = int(sfield.z[p,idz]/1000.0)
            # - Find min and max values. -------------------------------------------------------
            vmax = max(vmax, field[:,:,idz].max())
            vmin = min(vmin, field[:,:,idz].min())
            # - Make lats and lons. ------------------------------------------------------------
            lats = 90.0 - sfield.theta[p,:] * 180.0 / np.pi
            lons = sfield.phi[p,:] * 180.0 / np.pi
            lon, lat = np.meshgrid(lons, lats)
            # - Rotate if necessary. -----------------------------------------------------------
            if sfield.rotangle != 0.0:
                lat_rot = np.zeros(np.shape(lon),dtype=float)
                lon_rot = np.zeros(np.shape(lat),dtype=float)
                for idlon in np.arange(len(lons)):
                    for idlat in np.arange(len(lats)):
                        # lat_rot[idlat,idlon],lon_rot[idlat,idlon] = rotate_coordinates(sfield.n,-sfield.rotangle,90.0-lat[idlat,idlon],lon[idlat,idlon])
                        lat_rot[idlat,idlon],lon_rot[idlat,idlon]  = rotations.rotate_lat_lon(lat[idlat,idlon],lon[idlat,idlon], sfield.n,-sfield.rotangle)
                        lat_rot[idlat,idlon] = 90.0-lat_rot[idlat,idlon]
                lon = lon_rot
                lat = lat_rot
            cmap = colors.get_colormap('tomo_80_perc_linear_lightness')
            x, y = m(lon, lat)
            if pmaxdist > mindist:
                im = m.pcolormesh(x, y, field[:,:,idz]*1e9, shading='gouraud', cmap=cmap, vmin=valmin*1e9,vmax=valmax*1e9)
                # im = m.pcolormesh(x, y, field[:,:,idz]*1e9, shading='gouraud', cmap=cmap, vmin=valmin*1e9,vmax=valmax*1e9)
            else:
                im = m.pcolormesh(x, y, np.zeros(lon.shape), shading='gouraud', cmap=cmap, vmin=valmin,vmax=valmax)
    # - Add colobar and title. ------------------------------------------------------------------
    # cb = m.colorbar(im, "right", size="3%", pad='2%')
    # cb.ax.tick_params(labelsize=4) 
    try:
        geopolygons.PlotPolygon(inbasemap=m)
    except:
        pass
    # if component in UNIT_DICT:
    #     # cb.set_label(UNIT_DICT[component], fontsize=3, rotation=0)
    #     cb.set_label(r"$\frac{\mathrm{nm}}{\mathrm{s}}$", fontsize=3, rotation=0)
    # - Plot stations if available. ------------------------------------------------------------
    if (sfield.stations == True) & (stations==True):
        x,y = m(sfield.stlons,sfield.stlats)
        for n in range(sfield.n_stations):
            plt.text(x[n],y[n],sfield.stnames[n][:4])
            plt.plot(x[n],y[n],'ro')
    ################################################################################
    maxdist=iterN*0.05*3.
    Tfield.reset_reason(dist=maxdist)
    Afield.reset_reason(dist=maxdist)
    Afield.np2ma()
    Tfield.np2ma()
    plt.title('Wavefield', fontsize=10)
    ax2 = plt.subplot(gs[0, 1])
    # ax2=plt.subplot2grid((2,3),(0,1), rowspan=1, colspan=1)
    Tfield.plot_field(contour=True, showfig=False, vmin=0, vmax=1500., geopolygons=geopolygons)
    plt.title('Travel time', fontsize=10)
    ax3 = plt.subplot(gs[0, 2])
    Afield.plot_field(contour=False,showfig=False, vmin=0, vmax=1200., geopolygons=geopolygons)
    plt.title('Amplitude', fontsize=10)
    ax4 = plt.subplot(gs[1, 0])
    Tfield.plot_appV(showfig=False, geopolygons=geopolygons)
    plt.title('Apparent phase velocity', fontsize=10)
    ax5 = plt.subplot(gs[1, 1])
    Tfield.plot_diffa(showfig=False)
    plt.title('Great circle deflection', fontsize=10)
    ax6= plt.subplot(gs[1, 2])
    Afield.plot_lplcC(showfig=False, infield=Tfield, geopolygons=geopolygons)
    plt.title('Amplitude correction term', fontsize=10)
    plt.suptitle('t = '+str(iterN*0.05)+' sec', fontsize=10)
    plt.tight_layout(h_pad=0.)
    outfname=outdir+'/'+fprx+'_%06d.png' %(iterN)
    savefig(outfname, format='png', dpi=dpi)
    return

    