
import obspy

class StaInfo(object):
    """
    An object contains a station information several methods for station related analysis.
    ========================================================================
    General Parameters:
    stacode     - station name
    network    - network
    lon,lat       - position for station
    elevation   - elevation
    ========================================================================
    """
    def __init__(self, stacode=None, network='SES',  lat=None, lon=None, elevation=None ):

        self.stacode=stacode
        self.network=network
        self.lon=lon
        self.lat=lat
        self.elevation=elevation

    def get_contents(self):
        if self.stacode==None:
            print 'StaInfo NOT Initialized yet!'
            return
        print 'Network:%16s' %(self.network)
        print 'Station:%20s' %(self.stacode)
        print 'Longtitude:%17.3f' %(self.lon)
        print 'Latitude:  %17.3f' %(self.lat)
        return
    
    def GetPoslon(self):
        if self.lon<0:
            self.lon=self.lon+360.
        return
    
    def GetNeglon(self):
        if self.lon>180.:
            self.lon=self.lon-360.
        return
        
class StaLst(object):
    """
    An object contains a station list(a list of StaInfo object) information several methods for station list related analysis.
        stations: list of StaInfo
    """
    def __init__(self,stations=None):
        self.stations=[]
        if isinstance(stations, StaInfo):
            stations = [stations]
        if stations:
            self.stations.extend(stations)

    def __add__(self, other):
        """
        Add two StaLst with self += other.
        """
        if isinstance(other, StaInfo):
            other = StaLst([other])
        if not isinstance(other, StaLst):
            raise TypeError
        stations = self.stations + other.stations
        return self.__class__(stations=stations)

    def __len__(self):
        """
        Return the number of Traces in the StaLst object.
        """
        return len(self.stations)

    def __getitem__(self, index):
        """
        __getitem__ method of StaLst objects.
        :return: StaInfo objects
        """
        if isinstance(index, slice):
            return self.__class__(stations=self.stations.__getitem__(index))
        else:
            return self.stations.__getitem__(index)

    def append(self, station):
        """
        Append a single StaInfo object to the current StaLst object.
        """
        if isinstance(station, StaInfo):
            self.stations.append(station)
        else:
            msg = 'Append only supports a single StaInfo object as an argument.'
            raise TypeError(msg)
        return self

    def read(self, stafile):
        """
        Read Sation List from a txt file
        stacode longitude latidute network
        """
        with open(stafile, 'r') as f:
            Sta=[]
            f.readline()
            L=0
            for lines in f.readlines():
                L+=1
                if L%2 ==1:
                    lines=lines.split('.')
                    network=lines[0]
                    stacode=lines[1]
                if L%2 ==0:
                    lines=lines.split()
                    lat=90. - float(lines[0])
                    lon=float(lines[1])
                    elevation=float(lines[2])
                    netsta=network+'.'+stacode
                    if Sta.__contains__(netsta):
                        index=Sta.index(netsta)
                        if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                            raise ValueError('Incompatible Station Location:' + netsta+' in Station List!')
                        else:
                            print 'Warning: Repeated Station:' +netsta+' in Station List!'
                            continue
                    Sta.append(netsta)
                    self.append(StaInfo (stacode=stacode, network=network, lon=lon, lat=lat ))
        return

    def HomoStaLst(self, minlat, Nlat, minlon, Nlon, dlat, dlon, net='SES', prx='LF'):
        for ilon in xrange(Nlon):
            for ilat in xrange(Nlat):
                lon=minlon+ilon*dlon
                lat=minlat+ilat*dlat
                elevation=0.0
                stacode=prx+str(ilon)+'S'+str(ilat)
                self.stations.append(StaInfo (stacode=stacode, network=net, lon=lon, lat=lat))
                if ilon == Nlon -1 and ilat == Nlat -1:
                    print 'maxlat=', lat, 'maxlon=',lon
        return
    
    def write(self, outdir, eventnb=1):
        L=len(self.stations)
        outfname=outdir+'/recfile_%d' %eventnb
        with open(outfname,'wb') as f:
            f.writelines('%g\n' %L )
            for station in self.stations:
                staname=station.network+'.'+station.stacode+'.___'
                f.writelines('%s\n' %staname )
                f.writelines('%2.6f %3.6f 0.0\n' %(90.-station.lat, station.lon) )
        return
    
    def GetInventory(self, outfname=None, chans=['UP'], source='CU'):
        """
        Get obspy inventory, used for ASDF dataset
        ========================================================
        Input Parameters:
        outfname  - output stationxml file name (default = None, no output)
        chans        - channel list
        source       - source string
        Output:
        obspy.core.inventory.inventory.Inventory object, stationxml file(optional)
        ========================================================
        """
        stations=[]
        total_number_of_channels=len(chans)
        site=obspy.core.inventory.util.Site(name='01')
        creation_date=obspy.core.utcdatetime.UTCDateTime(0)
        for sta in self.stations:
            channels=[]
            for chan in chans:
                channel=obspy.core.inventory.channel.Channel(code=chan, location_code='01', latitude=sta.y/100000., longitude=sta.x/100000.,
                        elevation=sta.z, depth=0.0)
                channels.append(channel)
            station=obspy.core.inventory.station.Station(code=sta.stacode, latitude=sta.y/100000., longitude=sta.x/100000., elevation=sta.z,
                    site=site, channels=channels, total_number_of_channels = total_number_of_channels, creation_date = creation_date)
            stations.append(station)
        network=obspy.core.inventory.network.Network(code=sta.network, stations=stations)
        networks=[network]
        inv=obspy.core.inventory.inventory.Inventory(networks=networks, source=source)
        if outfname!=None:
            inv.write(outfname, format='stationxml')
        return inv
    