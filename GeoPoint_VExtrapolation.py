import numpy as np
import GeoPoint as GP


datadir='/lustre/janus_scratch/life9360/EA_model_smooth/ses3d_input_model'
mapfile='/projects/life9360/EA_MODEL/EA_grid_point.lst'
GeoLst=GP.GeoMap()
GeoLst.ReadGeoMapLst(mapfile)
GeoLst.SetVProfileFname(prefix='', suffix='_mod')
GeoLst.LoadVProfile(datadir=datadir, dirPFX=None)

ak135fname='/projects/life9360/code/ses3dPy/ak135.mod'
inak135Arr=np.loadtxt(ak135fname);
outdir='/lustre/janus_scratch/life9360/EA_model_smooth/ses3d_extended_model'
depth=np.array([10.,100.,200.])
dz=np.array([0.5,1.,2.])
depthInter=GP.GenerateDepthArr(depth, dz)

GeoLst.VProfileExtend(outdir, ak135fname=ak135fname, depthInter=depthInter, maxdepth=510)

