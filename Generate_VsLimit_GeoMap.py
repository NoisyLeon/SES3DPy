import SES3DPy
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
import GeoPoint as GP


datadir='/lustre/janus_scratch/life9360/EA_MODEL/ses3d_extended_model'
minlat=15.
maxlat=55.
minlon=70.
maxlon=150.

dlat=np.array([0.5,0.5,0.5,0.5,0.5])
dlon=dlat
depth=np.array([10., 100., 200., 410., 500.])
dz=np.array([0.5, 2., 2., 2., 5.])
ModelGen=SES3DPy.ses3dModelGen(minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon,\
    depth=depth, dlat=dlat, dlon=dlon, dz=dz)
ModelGen.GetBlockArrLst()
outdir='/lustre/janus_scratch/life9360/EA_MODEL/Check_ses3d_model_1km_sec_min_GeoMap'
# ModelGen.generateBlockFile(outdir=outdir)
# outdir='/lustre/janus_scratch/life9360/EA_MODEL/Check_ses3d_model_dV_1km_sec_min'
avgfname='/lustre/janus_scratch/life9360/EA_MODEL/ak135.mod'
ModelGen.generateVsLimitedGeoMap(datadir=datadir, outdir=outdir, Vsmin=1.0, avgfname=None, dataprx='', datasfx='_mod')

