import SES3DPy
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
import GeoPolygon as GeoPoly
stafile='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/station.lst_ses3d';
# datadir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/MODELS/MODELS_3D';
datadir='/projects/life9360/software/ses3d_r07_b/MODELS/MODELS'
mfname='dvsv'
# rotationfile='/projects/life9360/software/ses3d_r07_b/TOOLS/rotation_parameters.txt'
# setupfile='/projects/life9360/software/ses3d_r07_b/INPUT/setup'
# recfile='/projects/life9360/software/ses3d_r07_b/INPUT/recfile_1'
# sfield=SES3DPy.ses3d_fields(datadir, rotationfile, setupfile, recfile);

datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT'
rotationfile='/lustre/janus_scratch/life9360/SES3D_WorkingDir/TOOLS/rotation_parameters.txt'
setupfile='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT/setup'
recfile='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT/recfile_1'

# datadir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT_ak135_002'
# rotationfile='/lustre/janus_scratch/life9360/SES3D_WorkingDir/TOOLS/rotation_parameters.txt'
# setupfile='/lustre/janus_scratch/life9360/SES3D_WorkingDir/INPUT/setup'
# recfile='/lustre/janus_scratch/life9360/SES3D_WorkingDir/INPUT/recfile_1'

outdir='/lustre/janus_scratch/life9360/snapshots_ortho_filled';
sfield=SES3DPy.ses3d_fields(datadir, rotationfile, setupfile, recfile, field_type="velocity_snapshot");
mygeopolygons=GeoPoly.GeoPolygonLst();
mygeopolygons.ReadGeoPolygonLst('basin1');
sfield.plot_depth_slice('vz', 0.1, -2e-7, 2e-7, iteration=19000, verbose=True, stations=False, res="i", mapflag='regional_ortho',\
    mapfactor=2.5, geopolygons=mygeopolygons)

# sfield.plot_lat_depth_lon_slice('vz', 42, depth=10., minlon=130., maxlon=135., valmin=-5e-8, valmax=5e-8, iteration=6000)
# sfield.plot_colat_slice(component='vz', colat=49, valmin=-3e-8, valmax=3e-8, iteration=4600, verbose=True)
# sfield.MakeAnimation(component='vz', depth=50, valmin=-9e-8, valmax=9e-8, outdir=outdir,\
#         iter0=100, iterf=17100, dsnap=100, stations=False, res="i", mapflag='regional')
