import SES3DPy
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
import GeoPoint as GP
# 
# datadir='/lustre/janus_scratch/life9360/CHOOSE_V1';
# mapfile='/projects/life9360/EA_MODEL/EA_grid_point.lst';

datadir='/projects/life9360/InversionResult';
mapfile='/projects/life9360/EA_MODEL/Grid_point_sea.lst';

GeoLst=GP.GeoMap()
GeoLst.ReadGeoMapLst(mapfile)
GeoLst.SetVProfileFname( suffix='.acc.average.mod')
# GeoLst.LoadVProfile( datadir=datadir) # land
GeoLst.LoadVProfile(dirPFX='1_', datadir=datadir) # sea
GeoLst.GetMinVs();
# GeoLst.Vs2VpRho( flag=0, Vsmin=1.2 ); # land
GeoLst.Vs2VpRho( flag=1, Vsmin=1.2); # sea
outdir='/lustre/janus_scratch/life9360/EA_model_smooth/ses3d_input_model';
GeoLst.SetVProfileFname( prefix='', suffix='_mod')
# GeoLst.SaveVProfile( outdir=outdir, dirSFX=None)
GeoLst.ReplaceVProfile( outdir=outdir, dirSFX=None)

