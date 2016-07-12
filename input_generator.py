import stations
import events
from obspy.core.util.attribdict import AttribDict
import obspy.geodetics.base
import obspy
import os
import numpy as np
from lasif import rotations
import matplotlib.pyplot as plt
import warnings

class InputFileGenerator(object):
    """An object to generate input file for SES3D
    """
    def __init__(self):
        self.config = AttribDict()
        self.events = events.ses3dCatalog()
        self.stalst = stations.StaLst()
        # self.vmodel
        return
    
    def add_explosion(self, longitude, latitude, depth, m0):
        """
        Add explosion to catalog
        """
        self.events.add_explosion(longitude=longitude, latitude=latitude, depth=depth, m0=m0)
        self._check_time_step(evla = latitude, evlo=longitude)
        return
    
    def add_earthquake(self, longitude, latitude, depth, m_rr, m_tt, m_pp, m_tp, m_rt, m_rp):
        """
        Add earthquake to catalog
        """
        self.events.add_earthquake(longitude=longitude, latitude=latitude, depth=depth,
                           m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_tp=m_tp, m_rt=m_rt, m_rp=m_rp)
        self._check_time_step(evla = latitude, evlo=longitude)
        return
    
    def _check_time_step(self, evla, evlo, vmin = 3.0):
        minlat=self.config.mesh_min_latitude
        maxlat=self.config.mesh_max_latitude
        minlon=self.config.mesh_min_longitude 
        maxlon=self.config.mesh_max_longitude
        num_timpstep = self.config.number_of_time_steps
        dt = self.config.time_increment_in_s
        dist1, az, baz=obspy.geodetics.base.gps2dist_azimuth(evla, evlo, minlat, minlon) # distance is in m
        dist2, az, baz=obspy.geodetics.base.gps2dist_azimuth(evla, evlo, minlat, maxlon) # distance is in m
        dist3, az, baz=obspy.geodetics.base.gps2dist_azimuth(evla, evlo, maxlat, minlon) # distance is in m
        dist4, az, baz=obspy.geodetics.base.gps2dist_azimuth(evla, evlo, maxlat, maxlon) # distance is in m
        distArr = np.array([dist1/1000., dist2/1000., dist3/1000., dist4/1000.])
        distmax = distArr.max()
        if distmax/vmin * 1.5 > dt * num_timpstep:
            warnings.warn('Time length of seismogram is too short, recommended: '+
                        str(1.5*distmax/vmin)+' sec, actual: '+str(dt * num_timpstep)+' sec', UserWarning, stacklevel=1)
        return
        
    def set_config(self, num_timpstep, dt, nx_global, ny_global, nz_global, px, py, pz, minlat, maxlat, minlon, maxlon, zmin, zmax, 
                 isdiss=True, model_type=3, simulation_type=0, output_folder='../OUTPUT', adjoint_output_folder='../OUTPUT/ADJOINT',
                 lpd=4, displacement_snapshot_sampling=100, output_displacement=1, samp_ad=15 ):
        """ Set configuration for SES3D
        =============================================================
        Input Parameters:
        num_timpstep       - number of time step
        dt                            - time interval
        ----------------------------------------------------------------------------------------------------------------
        x : colatitude, y: longitude, z: depth 
        nx_global, ny_global, nz_global
                                        - number of finite elements in x/y/z direction  
        px, py, pz                - number of computational subdomain in x/y/z direction
        ----------------------------------------------------------------------------------------------------------------
        minlat, maxlat       - minimum/maximum latitude
        minlon, maxlon      - minimum/maximum longitude
        zmin, zmax             - minimum/maximum depth
        isdiss                      - dissipation on/off
        model_type             - 1 D Earth model type (default = 3)
                                            1. all-zero velocity and density model
                                            2. PREMiso
                                            3. all-zero velocity and density model with Q model QL6
                                            4. modified PREMiso with 220km discontinuity replaced
                                                by a linear gradient
                                            7. ak135
        simulation_type     - simulation type (default = 0)
                                            0: normal simulation; 1: adjoint forward; 2: adjoint reverse
        lpd                           - Lagrange polynomial degree (default = 4)
        samp_ad                 - sampling rate for adjoint field 
        =============================================================
        """
        print 'ATTENTION: Have You Updated the SOURCE/ses3d_conf.h and recompile the code???!!!'
        if simulation_type==0:
            self.config.simulation_type = "normal simulation"
        elif simulation_type==1:
            self.config.simulation_type = "adjoint forward"
        elif simulation_type==2:
            self.config.simulation_type = "adjoint reverse"
        # SES3D specific configuration  
        self.config.output_folder = output_folder
        self.config.lagrange_polynomial_degree = lpd
        # Time configuration.
        self.config.number_of_time_steps = num_timpstep
        self.config.time_increment_in_s = dt
        self.config.adjoint_forward_sampling_rate=samp_ad
        self.config.displacement_snapshot_sampling = displacement_snapshot_sampling
        self.config.output_displacement = output_displacement
        if nx_global%px!=0 or ny_global%py!=0 or  nz_global%pz!=0:
            raise ValueError('nx_global/px, ny_global/py, nz_global/pz must ALL be integer!')
        if int(nx_global/px)!=22 or int(ny_global/py)!=27 or int(nz_global/pz)!=7:
            print 'ATTENTION: elements in x/y/z direction per processor is NOT default Value! Check Carefully before running!'
        totalP=px*py*pz
        if totalP%12!=0:
            raise ValueError('Total number of processor must be 12N !')
        print '====================== Number of Nodes needed at Janus is: %g ======================' %(totalP/12)
        # SES3D specific discretization
        self.config.nx_global = nx_global
        self.config.ny_global = ny_global
        self.config.nz_global = nz_global
        self.config.px = px
        self.config.py = py
        self.config.pz = pz
        # Configure the mesh.
        self.config.mesh_min_latitude = minlat
        self.config.mesh_max_latitude = maxlat
        self.config.mesh_min_longitude = minlon 
        self.config.mesh_max_longitude = maxlon
        self.config.mesh_min_depth_in_km = zmin
        self.config.mesh_max_depth_in_km = zmax
        self.config.adjoint_forward_wavefield_output_folder = adjoint_output_folder
        self.model_type = model_type
        self.config.is_dissipative = isdiss
        self.CheckCFLCondition( )   
        # # Define the rotation. Take care this is defined as the rotation of the
        # # mesh.  The data will be rotated in the opposite direction! The following
        # # example will rotate the mesh 5 degrees southwards around the x-axis. For
        # # a definition of the coordinate system refer to the rotations.py file. The
        # # rotation is entirely optional.
        # gen.config.rotation_angle_in_degree = 5.0
        # gen.config.rotation_axis = [1.0, 0.0, 0.0]
        # Define Q
        return
    
    def CheckCFLCondition(self, C=0.35 ):
        """
        Check Courant-Frieddrichs-Lewy stability condition
        ======================================================================================
        Input Parameters:
        C                              - Courant number (default = 0.35, normally 0.3~0.4)
        ======================================================================================
        """
        if not os.path.isfile('./PREM.mod'):
            raise NameError('PREM Model File NOT exist!')
        InArr=np.loadtxt('./PREM.mod')
        depth=InArr[:,1]
        Vp=InArr[:,4]
        dt = self.config.time_increment_in_s 
        minlat=self.config.mesh_min_latitude
        maxlat=self.config.mesh_max_latitude
        minlon=self.config.mesh_min_longitude 
        maxlon=self.config.mesh_max_longitude
        nx_global = self.config.nx_global 
        ny_global = self.config.ny_global 
        nz_global = self.config.nz_global
        NGLL = self.config.lagrange_polynomial_degree
        zmin=self.config.mesh_min_depth_in_km
        zmax=self.config.mesh_max_depth_in_km
        maxabslat=max(abs(minlat), abs(maxlat))
        Vpmax=Vp[depth>zmax][0]
        dlat=(maxlat-minlat)/nx_global  # x : colatitude 
        dlon=(maxlon-minlon)/ny_global
        dz=(zmax-zmin)/nz_global
        distEW, az, baz=obspy.geodetics.base.gps2dist_azimuth(maxabslat, 45, maxabslat, 45+dlon) # distance is in m
        distEWmin=distEW/1000.*(6371.-zmax)/6371.
        distNS, az, baz=obspy.geodetics.base.gps2dist_azimuth(maxabslat, 45, maxabslat+dlat, 45) # distance is in m
        distNSmin=distNS/1000.*(6371.-zmax)/6371.
        dtEW=C*distEWmin/Vpmax/NGLL
        dtNS=C*distNSmin/Vpmax/NGLL
        dtZ=C*dz/Vpmax/NGLL
        print '=======================================================================' + \
                    '=========================================================================' 
        if dt > dtEW or dt > dtNS or dt > dtZ:
            raise ValueError('Time step violates Courant-Frieddrichs-Lewy Condition: ',dt, dtEW, dtNS, dtZ)
        else:
            print 'Time Step used:',dt, 'EW direction required:', dtEW, 'NS direction required:',dtNS, 'depth direction required:',dtZ
        print '=======================================================================' + \
                    '=========================================================================' 
        return
    
    def CheckMinWavelengthCondition(self, fmax=1.0/5.0, vmin=1.0, wpe=1.5):
        """
        Check minimum wavelength condition
        ==========================================================
        Input Parameters:
        fmax        - maximum frequency
        Vmin       - minimum velocity
        wpe         - wavelength per element
        ==========================================================
        """
        lamda=vmin/fmax
        NGLL = self.config.lagrange_polynomial_degree
        C=lamda*NGLL/5./wpe
        minlat=self.config.mesh_min_latitude
        maxlat=self.config.mesh_max_latitude
        minlon=self.config.mesh_min_longitude 
        maxlon=self.config.mesh_max_longitude
        minabslat=min(abs(minlat), abs(maxlat))
        mindep=self.config.mesh_min_depth_in_km
        maxdep=self.config.mesh_max_depth_in_km
        nx_global=self.config.nx_global
        ny_global=self.config.ny_global
        nz_global=self.config.nz_global
        dlat=(maxlat-minlat)/nx_global
        dlon=(maxlon-minlon)/ny_global
        dz=(maxdep-mindep)/nz_global
        distEW, az, baz=obspy.geodetics.base.gps2dist_azimuth(minabslat, 45, minabslat, 45+dlon) # distance is in m
        distNS, az, baz=obspy.geodetics.base.gps2dist_azimuth(minabslat, 45, minabslat+dlat, 45) # distance is in m
        distEW=distEW/1000.
        distNS=distNS/1000.
        print '=======================================================================' + \
                    '=========================================================================' 
        print 'Minimum wavelength condition number:',C, 'Element Size: dEW:',distEW, ' km, dNS: ', distNS,' km, dz:',dz
        if dz> C or distEW > C or distNS > C:
            raise ValueError('Minimum Wavelength Condition not satisfied!')
        print '=======================================================================' + \
            '=========================================================================' 
        return
    
    def add_stations(self, inSta):
        """
        Add station list
        """
        try:
            self.stalst= self.stalst + inSta
        except TypeError:
            SLst=stations.StaLst()
            SLst.read(inSta)
            self.stalst = self.stalst + SLst
        return
    
    def get_stf(self, stf, fmin=None, vmin = 1.0, fmax=None, plotflag=False):
        """
        Get source time function and filter it according to fmin/fmax
        ==========================================================
        Input Parameters:
        stf               - source time function
        fmin/fmax - minimum/maximum frequency
        plotflag      - plot source time function or not 
        ==========================================================
        """
        if self.config.time_increment_in_s!=stf.stats.delta or self.config.number_of_time_steps != stf.stats.npts:
            raise ValueError('Incompatible dt or npts in source time function!')
        if fmin !=None:
            stf.filter('highpass', freq=fmin, corners=4, zerophase=False)
        if fmax !=None:
            stf.filter('lowpass', freq=fmax, corners=5, zerophase=False)
        else:
            try:
                fmax=stf.fcenter*2.0 # should be 2.5
            except:
                raise AttributeError('Maximum frequency not specified!')
        if plotflag==True:
            stf.plotstf(fmax=fmax)
        self.CheckMinWavelengthCondition(fmax=fmax, vmin=vmin)
        self.config.source_time_function = stf.data
        return

    def write(self, outdir, verbose=True):
        """
        Write input files(setup, event_x, event_list, recfile_x, stf) to given directory
        """
        outdir=outdir
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        self.events.write(outdir, config=self.config)
        self.stalst.write(outdir)
        stf_fname=outdir+'/stf'
        try:
            np.savetxt(stf_fname, self.config.source_time_function)
        except:
            warnings.warn
        if self.config.is_dissipative and ( not os.path.isfile(outdir+'/relax') ):
            print outdir
            raise AttributeError('relax file not exists!')
        setup_file_template = (
            "MODEL ==============================================================="
            "================================================================="
            "=====\n"
            "{theta_min:<44.6f}! theta_min (colatitude) in degrees\n"
            "{theta_max:<44.6f}! theta_max (colatitude) in degrees\n"
            "{phi_min:<44.6f}! phi_min (longitude) in degrees\n"
            "{phi_max:<44.6f}! phi_max (longitude) in degrees\n"
            "{z_min:<44.6f}! z_min (radius) in m\n"
            "{z_max:<44.6f}! z_max (radius) in m\n"
            "{is_diss:<44d}! is_diss\n"
            "{model_type:<44d}! model_type\n"
            "COMPUTATIONAL SETUP (PARALLELISATION) ==============================="
            "================================================================="
            "=====\n"
            "{nx_global:<44d}! nx_global, "
            "(nx_global+px = global # elements in theta direction)\n"
            "{ny_global:<44d}! ny_global, "
            "(ny_global+py = global # elements in phi direction)\n"
            "{nz_global:<44d}! nz_global, "
            "(nz_global+pz = global # of elements in r direction)\n"
            "{lpd:<44d}! lpd, LAGRANGE polynomial degree\n"
            "{px:<44d}! px, processors in theta direction\n"
            "{py:<44d}! py, processors in phi direction\n"
            "{pz:<44d}! pz, processors in r direction\n"
            "ADJOINT PARAMETERS =================================================="
            "================================================================="
            "=====\n"
            "{adjoint_flag:<44d}! adjoint_flag (0=normal simulation, "
            "1=adjoint forward, 2=adjoint reverse)\n"
            "{samp_ad:<44d}! samp_ad, sampling rate of forward field\n"
            "{adjoint_wavefield_folder}")
        EARTH_RADIUS = 6371 * 1000.
        adjointdict={'normal simulation': 0, 'adjoint forward': 1, 'adjoint reverse': 2}
        setup_file = setup_file_template.format(
            # Colatitude! Swaps min and max.
            theta_min=rotations.lat2colat(float( self.config.mesh_max_latitude)),
            theta_max=rotations.lat2colat(float( self.config.mesh_min_latitude)),
            phi_min=float(self.config.mesh_min_longitude),
            phi_max=float(self.config.mesh_max_longitude),
            # Min/max radius and depth are inverse to each other.
            z_min=EARTH_RADIUS - (float(self.config.mesh_max_depth_in_km) * 1000.0),
            z_max=EARTH_RADIUS - (float(self.config.mesh_min_depth_in_km) * 1000.0),
            is_diss=1 if self.config.is_dissipative else 0,
            model_type=1,
            lpd=int(self.config.lagrange_polynomial_degree),
            # Computation setup.
            nx_global=self.config.nx_global,
            ny_global=self.config.ny_global,
            nz_global=self.config.nz_global,
            px=self.config.px,
            py=self.config.py,
            pz=self.config.pz,
            adjoint_flag=adjointdict[self.config.simulation_type],
            samp_ad=self.config.adjoint_forward_sampling_rate,
            adjoint_wavefield_folder=self.config.adjoint_forward_wavefield_output_folder)
        setup_fname=outdir+'/setup'
        with open(setup_fname, 'wb') as f:
            f.writelines(setup_file)

        
        
        

        
        
        
        
    