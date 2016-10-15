import symdata
import stations
import obspy
import obspy.signal.filter
import numpy as np
stafile='/work3/leon/ASDF_data/recfile_1'
dset = symdata.ses3dASDF('/work3/leon/ASDF_data/ses3d_2016.h5')

SLst=stations.StaLst()
SLst.read(stafile)

dset.write2sac(staid=None, lon=110, lat=34., outdir='.', SLst=SLst, channel='BXE')

