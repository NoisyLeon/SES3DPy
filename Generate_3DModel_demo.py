import SES3DPy
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
import GeoPoint as GP


# datadir='/lustre/janus_scratch/life9360/EA_MODEL/ses3d_final_extended_model'
datadir='/lustre/janus_scratch/life9360/EA_model_smooth/ses3d_final_extended_model'
# minlat=15.
# maxlat=55.
# minlon=70.
# maxlon=150.

minlat=25.
maxlat=52.
minlon=90.
maxlon=143.
# depth=np.array([10, 200, 500])

# minlat=30.
# maxlat=50.
# minlon=110.
# maxlon=140.
dlat=np.array([0.5,0.5,0.5,0.5,0.5])
dlon=dlat
depth=np.array([10., 100., 200., 410., 500.])
dz=np.array([0.5, 2., 2., 2., 5.])
ModelGen=SES3DPy.ses3dModelGen(minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon,\
    depth=depth, dlat=dlat, dlon=dlon, dz=dz)
ModelGen.GetBlockArrLst()
# outdir='/lustre/janus_scratch/life9360/EA_model_smooth'
# ModelGen.generateBlockFile(outdir=outdir)
outdir='/lustre/janus_scratch/life9360/EA_model_smooth/ses3d_block_model_AGU'
# avgfname='/lustre/janus_scratch/life9360/EA_MODEL/ak135.mod'
# ModelGen.generate3DModelFile(datadir=datadir, outdir=outdir, avgfname=avgfname, dataprx='', datasfx='_mod', Vsmin=1.0)

ModelGen.generateBlockFile(outdir=outdir)
ModelGen.generate3DModelFile(datadir=datadir, outdir=outdir, dataprx='', datasfx='_mod', Vsmin=1.2)


