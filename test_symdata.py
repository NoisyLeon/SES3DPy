import symdata
import stations
datadir = '/lustre/janus_scratch/life9360/ses3d_working_dir_2016/OUTPUT'
stafile='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT/recfile_1'
# dset = symdata.ses3dASDF('/lustre/janus_scratch/life9360/ASDF_data/ses3d_2016.h5')
# dset = symdata.ses3dASDF('/lustre/janus_scratch/life9360/ASDF_data/ses3d_2016_z_padding.h5')
dset = symdata.ses3dASDF('/lustre/janus_scratch/life9360/ASDF_data/ses3d_2016_10sec_20sec.h5')
# evlo=129.029
# evla=41.306
# # del dset.events
# dset.AddEvent(evlo, evla, evdp=1.0)
SLst=stations.StaLst()
SLst.read(stafile)

# st=dset.waveforms['SES.98S47'].ses3d_raw
# st.rotate(method='NE->RT', back_azimuth=59.30243969867691)
# startT=st[0].stats.starttime
# import obspy.signal.polarization

# dset.Readsac(datadir=datadir, minlon=90.25, Nlon=212, dlon=0.25, minlat=25., dlat=0.25,  Nlat=108, verbose=True)

# dset.get_wavefield(time=100., minlon=90.25, Nlon=212, dlon=0.25, minlat=25., dlat=0.25,  Nlat=108, net='EA')
# dset.readtxt(datadir=datadir, stafile=stafile, channel='all', verbose=True, VminPadding=2.7)
# dset.readtxt(datadir=datadir, stafile=stafile, channel='BXZ', verbose=True, VminPadding=2.7, factor=10)
# dset.decimate(10)

# dset.zero_padding(2.5)
# evlo=129.029
# evla=41.306
# # del dset.events
# dset.AddEvent(evlo, evla, evdp=1.0)
# # dset.pre_php(outdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016')
# dset.pre_php(outdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2015')