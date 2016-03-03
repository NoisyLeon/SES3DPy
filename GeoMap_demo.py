import GeoPoint as GP
import matplotlib.pyplot as plt

datadir='/lustre/janus_scratch/life9360/EA_MODEL/ses3d_extended_model'
outdir='/lustre/janus_scratch/life9360/EA_MODEL/ses3d_extended_model_001_fig'

datadir='/lustre/janus_scratch/life9360/EA_MODEL/Check_ses3d_model_1km_sec_min_GeoMap'
outdir='/lustre/janus_scratch/life9360/EA_MODEL/ses3d_extended_model_1km_sec_001_fig'
dirPFX='1_'
mapfile='/projects/life9360/EA_MODEL/EA_grid_point.lst';
minlat=30.
maxlat=50.
minlon=110.
maxlon=140.

GeoLst=GP.GeoMap()
GeoLst.ReadGeoMapLst(mapfile)
TrimGeoLst=GeoLst.Trim(maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat)
# GeoLst.SetAllfname()
TrimGeoLst.SetVProfileFname(prefix='', suffix='_mod')
TrimGeoLst.LoadVProfile(datadir=datadir, dirPFX=None)
# GeoLst.LoadGrDisp(datadir=datadir, dirPFX=dirPFX)
# GeoLst.LoadPhDisp(datadir=datadir, dirPFX=dirPFX)
TrimGeoLst.GetMapLimit()

# TrimGeoLst=GeoLst.Trim(maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat)
TrimGeoLst.BrowseFigures(datatype='VPr',outdir=outdir, dirPFX=None, depLimit=40, browseflag=False, saveflag=True)
# GeoLst.BrowseFigures(datatype='VPr',outdir=outdir, dirPFX=None, depLimit=40, browseflag=False, saveflag=True)