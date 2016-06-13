"""
An attempt to create a generic input file generator for different waveform
solvers.

Modified from Lion Krischer's wfs_input_generator 

"""

from lxml import etree
import copy
import fnmatch
import glob
import inspect
import io
import json
import obspy
from obspy import read_events
from obspy.core import AttribDict, read
from obspy.core.event import Event
from obspy.io.xseed import Parser
import os
import pkg_resources
import urllib2
import warnings
import stations

# Proper way to get the is_sac function.
is_sac = pkg_resources.load_entry_point(
    "obspy", "obspy.plugin.waveform.SAC", "isFormat")

def extract_coordinates_from_StationXML(file_or_file_object):
    root = etree.parse(file_or_file_object).getroot()
    namespace = root.nsmap[None]
    def _ns(tagname):
        return "{%s}%s" % (namespace, tagname)
    all_stations = []
    for network in root.findall(_ns("Network")):
        network_code = network.get("code")
        for station in network.findall(_ns("Station")):
            station_code = station.get("code")
            station_id = "%s.%s" % (network_code, station_code)
            station_latitude = _tag2obj(station, _ns("Latitude"), float)
            station_longitude = _tag2obj(station, _ns("Longitude"), float)
            station_elevation = _tag2obj(station, _ns("Elevation"), float)
            # Loop over all channels that might or might not be available. If
            # all channels have the same coordinates, use those. Otherwise use
            # the station coordinates.
            # Potential issues: The local depth is only stored at the channel
            # level and might thus not end up in the final dictionary.
            channel_coordinates = set()
            for channel in station.findall(_ns("Channel")):
                # Use a hashable dictionary to be able to use a set.
                coords = HashableDict(
                    latitude=_tag2obj(channel, _ns("Latitude"), float),
                    longitude=_tag2obj(channel, _ns("Longitude"), float),
                    elevation_in_m=_tag2obj(channel, _ns("Elevation"), float),
                    local_depth_in_m=_tag2obj(channel, _ns("Depth"), float))
                channel_coordinates.add(coords)
            # Check if it contains exactly one valid element.
            try:
                this_channel = channel_coordinates.pop()
                if len(channel_coordinates) != 0 or \
                        this_channel["latitude"] is None or \
                        this_channel["longitude"] is None or \
                        this_channel["elevation_in_m"] is None or \
                        this_channel["local_depth_in_m"] is None:
                    raise
                valid_channel = this_channel
            except:
                valid_channel = {
                    "latitude": station_latitude,
                    "longitude": station_longitude,
                    "elevation_in_m": station_elevation}
            valid_channel["id"] = station_id
            all_stations.append(valid_channel)
    return all_stations

class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.iteritems())))

def _tag2obj(element, tag, convert):
    """
    Helper function extracting and converting the text of any subelement..
    """
    try:
        return convert(element.find(tag).text)
    except:
        None

def unique_list(items):
    """
    Helper function taking a list of items and returning a list with duplicate
    items removed.
    """
    output = []
    for item in items:
        if item not in output:
            output.append(item)
    return output

class InputFileGenerator(object):
    """
    """
    def __init__(self):
        self.config = AttribDict()
        self._events = []
        self._stations = []
        self.__station_filter = None
        self.__event_filter = None

    def add_configuration(self, config):
        """
        ========================================================================
        Adds all items in config to the configuration.

        Useful for bulk configuration from external sources.

        :type config: dict or str
        :param config: Contains the new configuration items. Can be either a
            dictionary or a JSON document.
        ========================================================================
        """
        try:
            doc = json.loads(config)
        except:
            pass
        else:
            if isinstance(doc, dict):
                config = doc
        if not isinstance(config, dict):
            msg = "config must be either a dict or a single JSON document"
            raise ValueError(msg)

        self.config.__dict__.update(config)

    def add_events(self, events):
        """
        ========================================================================
        Add one or more events to the input file generator. Most inversions
        should specify only one event but some codes can deal with multiple
        events.

        Can currently deal with QuakeML files and obspy.core.event.Event
        objects.

        :type events: list or obspy.core.event.Catalog object
        :param events: A list of filenames, a list of obspy.core.event.Event
            objects, or an obspy.core.event.Catalog object.
        ========================================================================
        """
        # Try to interpret it as json. If it works and results in a list or
        # dicionary, use it!
        try:
            json_e = json.loads(events)
        except:
            pass
        else:
            # A simple string is also a valid JSON document.
            if isinstance(json_e, list) or isinstance(json_e, dict):
                events = json_e
        # Thin wrapper to enable single element treatment.
        if isinstance(events, Event) or isinstance(events, dict) or \
                not hasattr(events, "__iter__") or \
                (hasattr(events, "read") and
                 hasattr(events.read, "__call__")):
            events = [events, ]
        # Loop over all events.
        for event in events:
            # Download it if it is some kind of URL.
            if isinstance(event, basestring) and "://" in event:
                event = io.BytesIO(urllib2.urlopen(event).read())
            if isinstance(event, Event):
                self._parse_event(event)
                continue
            # If it is a dict do some checks and add it.
            elif isinstance(event, dict):
                required_keys = ["latitude", "longitude", "depth_in_km",
                                 "origin_time", "m_rr", "m_tt", "m_pp", "m_rt",
                                 "m_rp", "m_tp"]
                for key in required_keys:
                    if key not in event:
                        msg = (
                            "Each station events needs to at least have "
                            "{keys} keys.").format(
                            keys=", ".join(required_keys))
                        raise ValueError(msg)
                # Create new dict to not carry around any additional keys.
                ev = {
                    "latitude": float(event["latitude"]),
                    "longitude": float(event["longitude"]),
                    "depth_in_km": float(event["depth_in_km"]),
                    "origin_time": obspy.UTCDateTime(event["origin_time"]),
                    "m_rr": float(event["m_rr"]),
                    "m_tt": float(event["m_tt"]),
                    "m_pp": float(event["m_pp"]),
                    "m_rt": float(event["m_rt"]),
                    "m_rp": float(event["m_rp"]),
                    "m_tp": float(event["m_tp"])}
                if "description" in event and \
                        event["description"] is not None:
                    ev["description"] = str(event["description"])
                else:
                    ev["description"] = None
                self._events.append(ev)
                continue
            try:
                cat = read_events(event)
            except:
                pass
            else:
                for event in cat:
                    self._parse_event(event)
                continue
            msg = "Could not read %s." % event
            raise ValueError(msg)
        # Make sure each event is unique.
        self._events = unique_list(self._events)

    def add_stations(self, SLst):
        """
        ========================================================================
        Add the desired output stations to the input file generator.
        Can currently deal with SEED/XML-SEED files and dictionaries of the
        following form:
            {"latitude": 123.4,
             "longitude": 123.4,
             "elevation_in_m": 123.4,
             "local_depth_in_m": 123.4,
             "id": "network_code.station_code"}

        `local_depth_in_m` is optional and will be assumed to be zero if not
        present. It denotes the burrial of the sensor beneath the surface.

        If it is a SEED/XML-SEED files, all stations in it will be added.

        :type stations: List of filenames, list of dictionaries or a single
            filename, single dictionary.
        :param stations: The stations for which output files should be
            generated.
        ========================================================================
        """
        self.stalst=SLst
        
    @property
    def _filtered_stations(self):
        if not self.station_filter:
            return self._stations
        def filt(station):
            for pattern in self.station_filter:
                if fnmatch.fnmatch(station["id"], pattern):
                    return True
            return False
        return filter(filt, self._stations)

    @property
    def station_filter(self):
        return self.__station_filter

    @station_filter.setter
    def station_filter(self, value):
        try:
            value = json.loads(value)
        except:
            pass

        if not hasattr(value, "__iter__") and value is not None:
            msg = "Needs to be a list or other iterable."
            raise TypeError(msg)
        self.__station_filter = value

    @property
    def _filtered_events(self):
        if not self.event_filter:
            return self._events
        def filt(event):
            # No id will remove the event.
            if "_event_id" not in event:
                return False
            for event_id in self.event_filter:
                if event["_event_id"].lower() == event_id.lower():
                    return True
            return False
        return filter(filt, self._events)

    @property
    def event_filter(self):
        return self.__event_filter

    @event_filter.setter
    def event_filter(self, value):
        try:
            value = json.loads(value)
        except:
            pass
        if not hasattr(value, "__iter__") and value is not None:
            msg = "Needs to be a list or other iterable."
            raise TypeError(msg)
        self.__event_filter = value

    def _parse_seed(self, station_item, all_stations):
        """
        Helper function to parse SEED and XSEED files.
        """
        parser = Parser(station_item)
        for station in parser.stations:
            network_code = None
            station_code = None
            latitude = None
            longitude = None
            elevation = None
            local_depth = None
            for blockette in station:
                if blockette.id not in [50, 52]:
                    continue
                elif blockette.id == 50:
                    network_code = str(blockette.network_code)
                    station_code = str(blockette.station_call_letters)
                    continue
                elif blockette.id == 52:
                    latitude = blockette.latitude
                    longitude = blockette.longitude
                    elevation = blockette.elevation
                    local_depth = blockette.local_depth
                    break
            if None in [network_code, station_code, latitude, longitude,
                        elevation, local_depth]:
                msg = "Could not parse %s" % station_item
                raise ValueError(msg)
            stat = {
                "id": "%s.%s" % (network_code, station_code),
                "latitude": latitude,
                "longitude": longitude,
                "elevation_in_m": elevation,
                "local_depth_in_m": local_depth}
            if stat["id"] in all_stations:
                all_stations[stat["id"]].update(stat)
            else:
                all_stations[stat["id"]] = stat

    def write(self, output_dir=None):
        """
        ========================================================================
        Write an input file with the specified format.

        :param format: The requested format of the generated input files. Get a
            list of available format with a call to
            self.get_available_formats().
        :type output_dir: string
        :param output_dir: The folder where all files will be written to. If
            it does not exists, it will be created. Any files already in
            existence WILL be overwritten. So be careful.
        ========================================================================
        """
        # Check if the corresponding write function exists.
        self.__find_write_scripts()
        if format not in list(self.__write_functions.keys()):
            msg = "Format %s not found. Available formats: %s." % (
                format, list(self.__write_functions.keys()))
            raise ValueError(msg)
        # Make sure only unique stations and events are passed on. Sort
        # stations by id.
        _stations = copy.deepcopy(sorted(unique_list(self._filtered_stations),
                                         key=lambda x: x["id"]))
        _events = copy.deepcopy(unique_list(self._filtered_events))
        # Remove the "_event_id"s everywhere
        for event in _events:
            try:
                del event["_event_id"]
            except:
                pass
        # Set the correct write function.
        writer = self.__write_functions[format]
        config = copy.deepcopy(self.config)
        # Check that all required configuration values exist and convert to
        # the correct type.
        for config_name, value in writer["required_config"].iteritems():
            convert_fct, _ = value
            if config_name not in config:
                msg = ("The input file generator for '%s' requires the "
                       "configuration item '%s'.") % (format, config_name)
                raise ValueError(msg)
            try:
                config[config_name] = convert_fct(config[config_name])
            except:
                msg = ("The configuration value '%s' could not be converted "
                       "to '%s'") % (config_name, str(convert_fct))
                raise ValueError(msg)
        # Now set the optional and default parameters.
        for config_name, value in writer["default_config"].iteritems():
            default_value, convert_fct, _ = value
            if config_name in config:
                default_value = config[config_name]
            try:
                config[config_name] = convert_fct(default_value)
            except:
                msg = ("The configuration value '%s' could not be converted "
                       "to '%s'") % (config_name, str(convert_fct))
                raise ValueError(msg)
        # Call the write function. The write function is supposed to raise the
        # appropriate error in case anything is amiss.
        input_files = writer["function"](config=config, events=_events,
                                         stations=_stations)
        # If an output directory is given, it will be used.
        if output_dir:
            # Create the folder if it does not exist.
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.isdir(output_dir):
                msg = "output_dir %s is not a directory" % output_dir
                raise ValueError(msg)
            output_dir = os.path.abspath(output_dir)
            # Now loop over all files stored in the dictionary and write them.
            for filename, content in input_files.iteritems():
                with open(os.path.join(output_dir, filename), "wt") \
                        as open_file:
                    open_file.write(content)
        return input_files

    def __find_write_scripts(self):
        """
        Helper method to find all available writer scripts. A write script is
        defined as being in the folder "writer" and having a name of the form
        "write_XXX.py". It furthermore needs to have a write() method.
        """
        # Most generic way to get the 'backends' subdirectory.
        write_dir = os.path.join(os.path.dirname(inspect.getfile(
            inspect.currentframe())), "backends")
        files = glob.glob(os.path.join(write_dir, "write_*.py"))
        import_names = [os.path.splitext(os.path.basename(_i))[0]
                        for _i in files]
        write_functions = {}
        for name in import_names:
            module_name = "backends.%s" % name
            try:
                module = __import__(
                    module_name, globals(), locals(),
                    ["write", "REQUIRED_CONFIGURATION",
                     "DEFAULT_CONFIGURATION"], -1)
                function = module.write
                required_config = module.REQUIRED_CONFIGURATION
                default_config = module.DEFAULT_CONFIGURATION
            except Exception as e:
                print("Warning: Could not import %s." % module_name)
                print("\t%s: %s" % (e.__class__.__name__, str(e)))
                continue
            if not hasattr(function, "__call__"):
                msg = "Warning: write in %s is not a function." % module_name
                print(msg)
                continue
            # Append the function and some more parameters.
            write_functions[name[6:]] = {
                "function": function,
                "required_config": required_config,
                "default_config": default_config}
        self.__write_functions = write_functions

    def get_available_formats(self):
        """
        Get a list of all available formats.
        """
        self.__find_write_scripts()
        return list(self.__write_functions.keys())

    def get_config_params(self, solver_name):
        self.__find_write_scripts()
        if solver_name not in self.__write_functions.keys():
            msg = "Solver '%s' not found." % solver_name
            raise ValueError(msg)
        writer = self.__write_functions[solver_name]

        return writer["required_config"], writer["default_config"]

    def _parse_event(self, event):
        """
        Check and parse events.

        Each event at least needs to have an origin and a moment tensor,
        otherwise an error will be raised.
        """
        # Do a lot of checks first to be able to give descriptive error
        # messages.
        if not event.origins:
            msg = "Each event needs to have an origin."
            raise ValueError(msg)
        if not event.focal_mechanisms:
            msg = "Each event needs to have a focal mechanism."
            raise ValueError(msg)
        # Choose either the preferred origin or the first one.
        origin = event.preferred_origin() or event.origins[0]
        # Same with the focal mechanism.
        foc_mec = event.preferred_focal_mechanism() or \
            event.focal_mechanisms[0]
        # Origin needs to have latitude, longitude, depth and time
        if None in (origin.latitude, origin.longitude, origin.depth,
                    origin.time):
            msg = ("Every event origin needs to have latitude, longitude, "
                   "depth and time")
            raise ValueError(msg)
        # The focal mechanism of course needs to have a moment tensor.
        if not foc_mec.moment_tensor or not foc_mec.moment_tensor.tensor:
            msg = "Every event needs to have a moment tensor."
            raise ValueError(msg)
        # Also all six components need to be specified.
        mt = foc_mec.moment_tensor.tensor
        if None in (mt.m_rr, mt.m_tt, mt.m_pp, mt.m_rt, mt.m_rp, mt.m_tp):
            msg = "Every event needs all six moment tensor components."
            raise ValueError(msg)
        # Extract event descriptions.
        if event.event_descriptions:
            description = ", ".join(i.text for i in event.event_descriptions)
        else:
            description = None
        # Now the event should be valid.
        self._events.append({
            "latitude": origin.latitude,
            "longitude": origin.longitude,
            "depth_in_km": origin.depth / 1000.0,
            "origin_time": origin.time,
            "m_rr": mt.m_rr,
            "m_tt": mt.m_tt,
            "m_pp": mt.m_pp,
            "m_rt": mt.m_rt,
            "m_rp": mt.m_rp,
            "m_tp": mt.m_tp,
            "_event_id": event.resource_id.resource_id,
            "description": description})
        
         
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
    
    def SetConfig(self, num_timpstep, dt, nx_global, ny_global, nz_global, px, py, pz, minlat, maxlat, minlon, maxlon, mindep, maxdep, \
            isdiss=True, SimType=0, OutFolder="../OUTPUT", ):
        
        print 'ATTENTION: Have You Updated the SOURCE/ses3d_conf.h and recompile the code???!!!'
        if SimType==0:
            self.config.simulation_type = "normal simulation"
        elif SimType==1:
            self.config.simulation_type = "adjoint forward"
        elif SimType==2:
            self.config.simulation_type = "adjoint reverse"
        # SES3D specific configuration  
        self.config.output_folder = OutFolder
    
        # Time configuration.
        self.config.number_of_time_steps = num_timpstep
        self.config.time_increment_in_s = dt
        if (nx_global/px-int(nx_global/px))!=0.0 or (ny_global/py-int(ny_global/py))!=0.0 or (nz_global/pz-int(nz_global/pz))!=0.0:
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
        self.config.mesh_min_depth_in_km = mindep
        self.config.mesh_max_depth_in_km = maxdep
        self.CheckCFLCondition(minlat, maxlat, minlon, maxlon, mindep, maxdep, nx_global, ny_global, nz_global, dt)
        
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
    
    def CheckCFLCondition(self, minlat, maxlat, minlon, maxlon, mindep, maxdep, nx_global, ny_global, nz_global, dt, NGLL=4, C=0.35 ):
        if not os.path.isfile('./PREM.mod'):
            raise NameError('PREM Model File NOT exist!')
        InArr=np.loadtxt('./PREM.mod')
        depth=InArr[:,1]
        Vp=InArr[:,4]
        Vpmax=Vp[depth>maxdep][0]
        maxabslat=max(abs(minlat), abs(maxlat))
        dlat=(maxlat-minlat)/nx_global
        dlon=(maxlon-minlon)/ny_global
        dz=(maxdep-mindep)/nz_global
        distEW, az, baz=obsGeo.gps2DistAzimuth(maxabslat, 45, maxabslat, 45+dlon) # distance is in m
        distEWmin=distEW/1000.*(6371.-maxdep)/6371.
        distNS, az, baz=obsGeo.gps2DistAzimuth(maxabslat, 45, maxabslat+dlat, 45) # distance is in m
        distNSmin=distNS/1000.*(6371.-maxdep)/6371.
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
        distEW, az, baz=obsGeo.gps2DistAzimuth(minabslat, 45, minabslat, 45+dlon) # distance is in m
        distNS, az, baz=obsGeo.gps2DistAzimuth(minabslat, 45, minabslat+dlat, 45) # distance is in m
        distEW=distEW/1000.
        distNS=distNS/1000.
        print 'Condition number:',C, 'Element Size: dEW:',distEW, ' km, dNS: ', distNS,' km, dz:',dz
        if dz> C or distEW > C or distNS > C:
            raise ValueError('Minimum Wavelength Condition not satisfied!')
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
