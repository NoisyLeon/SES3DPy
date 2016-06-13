import stations
import events
from obspy.core.util.attribdict import AttribDict
import obspy.geodetics.base
import obspy



class InputFileGenerator(object):
    """
    """
    def __init__(self):
        self.config = AttribDict()
        self.events = events.ses3dCatalog()
        self.stalst = stations.StaLst()
        # self.vmodel
        return
    
    def add_explosion(self, longitude, latitude, depth, m0):
        self.events.add_explosion(longitude=longitude, latitude=latitude, depth=depth, m0=m0)
        return
    
    def add_earthquake(self, longitude, latitude, depth, m_rr, m_tt, m_pp, m_tp, m_rt, m_rp):
        self.events.add_earthquake(longitude=longitude, latitude=latitude, depth=depth,
                           m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_tp=m_tp, m_rt=m_rt, m_rp=m_rp)
        return
    
    def setconfig(self, num_timpstep, dt, nx_global, ny_global, nz_global, px, py, pz,
            minlat, maxlat, minlon, maxlon, zmin, zmax, \
            isdiss=True, simulation_type=0, OutFolder='../OUTPUT', lpd=4,  ):
        
        print 'ATTENTION: Have You Updated the SOURCE/ses3d_conf.h and recompile the code???!!!'
        if simulation_type==0:
            self.config.simulation_type = "normal simulation"
        elif simulation_type==1:
            self.config.simulation_type = "adjoint forward"
        elif simulation_type==2:
            self.config.simulation_type = "adjoint reverse"
        # SES3D specific configuration  
        self.config.output_folder = OutFolder
    
        # Time configuration.
        self.config.number_of_time_steps = num_timpstep
        self.config.time_increment_in_s = dt
        if nx_global%px!=0 or ny_global%py!=0 or  nz_global%pz!=0:
            raise ValueError('nx_global/px, ny_global/py, nz_global/pz must ALL be integer!')
        if int(nx_global/px)!=22 or int(ny_global/py)!=27 or int(nz_global/pz)!=7:
            print 'ATTENTION: elements in x/y/z direction per processor is NOT default Value! Check Carefully before running!'
        totalP=px*py*pz
        if totalP%12!=0:
            raise ValueError('total number of processor must be 12N !')
        print 'Number of Nodes needed at Janus is: %g' %(totalP/12)
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
        self.CheckCFLCondition(minlat, maxlat, minlon, maxlon, zmin, zmax, nx_global, ny_global, nz_global, dt)   
        # # Define the rotation. Take care this is defined as the rotation of the
        # # mesh.  The data will be rotated in the opposite direction! The following
        # # example will rotate the mesh 5 degrees southwards around the x-axis. For
        # # a definition of the coordinate system refer to the rotations.py file. The
        # # rotation is entirely optional.
        # gen.config.rotation_angle_in_degree = 5.0
        # gen.config.rotation_axis = [1.0, 0.0, 0.0]
        # Define Q
        self.config.is_dissipative = isdiss
        
        return
    
    def CheckCFLCondition(self, minlat, maxlat, minlon, maxlon, zmin, zmax, nx_global, ny_global, nz_global, dt, NGLL=4, C=0.35 ):
        if not os.path.isfile('./PREM.mod'):
            raise NameError('PREM Model File NOT exist!')
        InArr=np.loadtxt('./PREM.mod')
        depth=InArr[:,1]
        Vp=InArr[:,4]
        Vpmax=Vp[depth>zmax][0]
        maxabslat=max(abs(minlat), abs(maxlat))
        dlat=(maxlat-minlat)/nx_global
        dlon=(maxlon-minlon)/ny_global
        dz=(zmax-zmin)/nz_global
        distEW, az, baz=obspy.geodetics.base.gps2dist_azimuth(maxabslat, 45, maxabslat, 45+dlon) # distance is in m
        distEWmin=distEW/1000.*(6371.-zmax)/6371.
        distNS, az, baz=obspy.geodetics.base.gps2dist_azimuth(maxabslat, 45, maxabslat+dlat, 45) # distance is in m
        distNSmin=distNS/1000.*(6371.-zmax)/6371.
        dzmin=dz
        dtEW=C*distEWmin/Vpmax/NGLL
        dtNS=C*distNSmin/Vpmax/NGLL
        dtZ=C*dz/Vpmax/NGLL
        print Vpmax, distEWmin, distNSmin
        if dt > dtEW or dt > dtNS or dt > dtZ:
            raise ValueError('Time step violates Courant-Frieddrichs-Lewy Condition: ',dt, dtEW, dtNS, dtZ)
        else:
            print 'Time Step: ',dt, dtEW, dtNS, dtZ
        return
    
    def CheckMinWavelengthCondition(self, NGLL=4., fmax=1.0/5.0, Vmin=1.0):
        lamda=Vmin/fmax
        C=NGLL/5.*lamda
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
        print 'Condition number:',C, 'Element Size: dEW:',distEW, ' km, dNS: ', distNS,' km, dz:',dz
        if dz> C or distEW > C or distNS > C:
            raise ValueError('Minimum Wavelength Condition not satisfied!')
        return
    
    def add_stations(self, SLst):
        self.stalst.append(SLst)
        return
        
         
    def AddExplosion(self, lon, lat, depth, mag, magtype='moment', des='NorthKorea', origintime=obspy.UTCDateTime()):
        if magtype=='moment':
            M0=10**(1.5*mag+9.1)
            Miso=M0*math.sqrt(2./3.)
        temp_event={"latitude": lat,"longitude": lon, "depth_in_km": depth,
            "origin_time": origintime, "description": des,\
            "m_rr": Miso, "m_tt": Miso, "m_pp": Miso,\
            "m_rt": 0.0, "m_rp": 0.0, "m_tp": 0.0}
        self.add_events(temp_event)
        return
    
    def GetRelax(self, datadir='', QrelaxT=[], Qweight=[], fmin=None, fmax=None):
        infname_tau=datadir+'/LF_RELAX_tau'
        infname_D=datadir+'/LF_RELAX_D'
        
        if (len(QrelaxT)==0 or len(Qweight)==0) and\
            ( not (os.path.isfile(infname_tau) and os.path.isfile(infname_D)) ):
            if fmin==None or fmax==None:
                raise ValueError('Error Input!')
                print 'Computing relaxation times!'
                Qmodel=Q_model( fmin=fmin, fmax=fmax )
                Qmodel.Qdiscrete()
                QrelaxT=Qmodel.tau_s.tolist()
                Qweight=Qmodel.D.tolist()
                Qmodel.PlotQdiscrete()
        if os.path.isfile(infname_tau) and os.path.isfile(infname_D):
            print 'Reading relaxation time from LF_RELAX_* !'
            QrelaxT=np.loadtxt(infname_tau)
            Qweight=np.loadtxt(infname_D)
        self.config.Q_model_relaxation_times = QrelaxT
        self.config.Q_model_weights_of_relaxation_mechanisms = Qweight
        
        return
    
    
    
    def get_stf(self, stf, fmin, fmax, plotflag=True):
        if self.config.time_increment_in_s!=stf.stats.delta or self.config.number_of_time_steps != stf.stats.npts:
            raise ValueError('Incompatible dt or npts in source time function!')
        stf.filter('highpass', freq=fmin, corners=4, zerophase=False)
        stf.filter('lowpass', freq=fmax, corners=5, zerophase=False)
        if plotflag==True:
            ax=plt.subplot(411)
            ax.plot(np.arange(stf.stats.npts)*stf.stats.delta, stf.data, 'k-', lw=3)
            plt.xlim(0,2./fmin)
            plt.xlabel('time [s]')
            plt.title('source time function (time domain)')
            ax=plt.subplot(412)
            stf.dofft()
            stf.plotfreq()
            plt.xlim(fmin/5.,fmax*5.)
            plt.xlabel('frequency [Hz]')
            plt.title('source time function (frequency domain)')
            
            ax=plt.subplot(413)
            outSTF=stf.copy()
            stf.differentiate()
            ax.plot(np.arange(stf.stats.npts)*stf.stats.delta, stf.data, 'k-', lw=3)
            plt.xlim(0,2./fmin)
            plt.xlabel('time [s]')
            plt.title('Derivative of source time function (time domain)')
            
            ax=plt.subplot(414)
            stf.dofft()
            stf.plotfreq()
            plt.xlim(fmin/5.,fmax*5.)
            plt.xlabel('frequency [Hz]')
            plt.title('Derivative of source time function (frequency domain)')
            
            plt.show()
        # self.CheckMinWavelengthCondition(fmax=fmax)
        self.config.source_time_function = outSTF.data
        return
        
    def make_stf(self, dt=0.10, nt=5000, fmin=1.0/100.0, fmax=1.0/5.0, plotflag=False):
        """
        Generate a source time function for ses3d by applying a bandpass filter to a Heaviside function.
    
        make_stf(dt=0.13, nt=4000, fmin=1.0/100.0, fmax=1.0/8.0, plotflag=False)
    
        dt: Length of the time step. Must equal dt in the event_* file.
        nt: Number of time steps. Must equal to or greater than nt in the event_* file.
        fmin: Minimum frequency of the bandpass.
        fmax: Maximum frequency of the bandpass.
        """
        #- Make time axis and original Heaviside function. --------------------------------------------
        t = np.arange(0.0,float(nt+1)*dt,dt)
        h = np.ones(len(t))
        #- Apply filters. -----------------------------------------------------------------------------
        h = flt.highpass(h, fmin, 1.0/dt, 3, zerophase=False)
        h = flt.lowpass(h, fmax, 1.0/dt, 6, zerophase=False)
        # h=GaussianFilter(h , fcenter=fmax, df=1.0/dt, fhlen=0.008)
        # h=flt.bandstop(h, freqmin=fmin, freqmax=fmax, df=1.0/dt, corners=4, zerophase=True)
    
        #- Plot output. -------------------------------------------------------------------------------
        if plotflag == True:
            #- Time domain.
            plt.plot(t,h,'k')
            plt.xlim(0.0,float(nt)*dt)
            plt.xlabel('time [s]')
            plt.title('source time function (time domain)')
            plt.show()
            #- Frequency domain.
            hf = np.fft.fft(h)
            f = np.fft.fftfreq(len(hf), dt)
            plt.semilogx(f,np.abs(hf),'k')
            plt.plot([fmin,fmin],[0.0, np.max(np.abs(hf))],'r--')
            plt.text(1.1*fmin, 0.5*np.max(np.abs(hf)), 'fmin')
            plt.plot([fmax,fmax],[0.0, np.max(np.abs(hf))],'r--')
            plt.text(1.1*fmax, 0.5*np.max(np.abs(hf)), 'fmax')
            plt.xlim(0.1*fmin,10.0*fmax)
            plt.xlabel('frequency [Hz]')
            plt.title('source time function (frequency domain)')
            plt.show()
        # np.savetxt('mystf', h, fmt='%g')
        self.config.source_time_function = h
        print 'Maximum frequency is doubled to apprximate stop frequency!'
        self.CheckMinWavelengthCondition(fmax=fmax)
        return
    
    def WriteSES3D(self, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        self.write(format="ses3d_4_1", output_dir=outdir)
        return

    def GenerateReceiverLst(self, minlat, maxlat, minlon, maxlon, dlat, dlon, PRX='LF'):
        LonLst=np.arange(int((maxlon-minlon)/dlon)+1)*dlon+minlon
        LatLst=np.arange(int((maxlat-minlat)/dlat)+1)*dlat+minlat
        L=0
        for lon in LonLst:
            for lat in LatLst:
                strlat='%g' %(lat*100)
                strlon='%g' %(lon*100)
                stacode=PRX+str(strlat)+'S'+str(strlon)
                net=PRX
                lon=lon
                lat=lat
                print 'Adding Station: ', stacode
                elevation=0.0
                temp_sta={ "id": net+'.'+stacode, "latitude": lat, "longitude": lon, "elevation_in_m": elevation }
                self.add_stations(temp_sta)
                L=L+1
        if (L>800):
            print 'ATTENTION: Number of receivers larger than default value!!! Check Carefully before running!'
        return