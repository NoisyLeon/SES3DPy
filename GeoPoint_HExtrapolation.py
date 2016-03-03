import SES3DPy
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
import GeoPoint as GP
datadir='/lustre/janus_scratch/life9360/EA_model_smooth/ses3d_extended_model'
mapfile='/projects/life9360/EA_MODEL/EA_grid_point.lst'
GeoLst=GP.GeoMap()
GeoLst.ReadGeoMapLst(mapfile)
GeoLst.SetVProfileFname(prefix='', suffix='_mod')
GeoLst.LoadVProfile(datadir=datadir, dirPFX=None)
GeoLst.GetMinVs();
avgEAfname='/lustre/janus_scratch/life9360/EA_MODEL/ses3d_extended_model/EAsia_avg.mod';
# refPfname='/lustre/janus_scratch/life9360/EA_MODEL/ses3d_extended_model/115_48_mod'
# outdir='/lustre/janus_scratch/life9360/East_Asia_ses3d/ses3d_SH_extended_model_Check'
outdir='/lustre/janus_scratch/life9360/EA_model_smooth/ses3d_final_extended_model'
minlat=15.
maxlat=55.
minlon=70.
maxlon=150.

# minlatM=19.
# maxlatM=52.5
# minlonM=75.
# maxlonM=143.5

GeoLst.HExtrapolationNearestParallel(avgfname=avgEAfname, outdir=outdir, dlat=0.5, dlon=0.5,\
    maxlat=maxlat, minlat=minlat, maxlon=maxlon, minlon=minlon)


