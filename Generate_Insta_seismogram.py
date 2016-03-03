import obspy
import instaseis
import obspy.core.util.geodetics as obsGeo
# evlo=129.02;
# evla=41.31;
# stlo=113.00;
# stla=54.00;

evlo=0;
evla=0;
stlo=20;
stla=0;

evdp=1;
stacode="S001"
db = instaseis.open_db("/projects/life9360/instaseis_seismogram/10s_PREM_ANI_FORCES")
receiver = instaseis.Receiver(latitude=stla, longitude=stlo, network="LF",station=stacode)
source = instaseis.Source(
    latitude=evla, longitude=evlo, depth_in_m=1000.*evdp,
    m_rr = 1.451959e+15,
    m_tt = 1.451959e+15,
    m_pp=1.451959e+15,
    m_rt=0,
    m_rp=0,
    m_tp=0,
    origin_time=obspy.UTCDateTime(2011, 1, 2, 3, 4, 5));


st = db.get_seismograms(source=source, receiver=receiver);

tr_z=st[0];
tr_z.stats['sac']={};
tr_z.stats.sac.evlo=evlo;
tr_z.stats.sac.evla=evla;
tr_z.stats.sac.stlo=stlo;
tr_z.stats.sac.stla=stla;
tr_z.stats.sac.evdp=evdp;

dist, az, baz=obsGeo.gps2DistAzimuth(evla, evlo, stla, stlo); # distance is in m
tr_z.stats.sac.dist=dist/1000.;

tr_z.write('/projects/life9360/instaseis_seismogram/'+stacode+'_INSTASEIS.LXZ.SAC',format='sac')
