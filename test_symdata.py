import symdata
datadir = '/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_SAC'
dset = symdata.ses3dASDF('/lustre/janus_scratch/life9360/ASDF_data/ses3d.h5')
# dset.Readsac(datadir=datadir, minlon=90.25, Nlon=212, dlon=0.25, minlat=25., dlat=0.25,  Nlat=108, verbose=True)

dset.get_wavefield(time=100., minlon=90.25, Nlon=212, dlon=0.25, minlat=25., dlat=0.25,  Nlat=108, net='EA')