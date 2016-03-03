import SES3DPy
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
import GeoPolygon as GeoPoly

datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT'
rotationfile='/lustre/janus_scratch/life9360/SES3D_WorkingDir/TOOLS/rotation_parameters.txt'
setupfile='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT/setup'
recfile='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT/recfile_1'
outdir='/lustre/janus_scratch/life9360/snapshots_ortho_EA_AGU_001_100_png_basin_no_axis_reset_y';

evlo=129.029;
evla=41.306;
mygeopolygons=GeoPoly.GeoPolygonLst();
mygeopolygons.ReadGeoPolygonLst('basin1');
sfield=SES3DPy.ses3d_fields(datadir, rotationfile, setupfile, recfile, field_type="velocity_snapshot");
# sfield.plot_colat_slice(component='vz', colat=49, valmin=-3e-8, valmax=3e-8, iteration=4600, verbose=True)
sfield.MakeAnimationParallel(evlo=evlo, evla=evla, component='vz', depth=0.1, valmin=-1.5e-7, valmax=1.5e-7, outdir=outdir,\
        iter0=18000, iterf=20000, dsnap=200, stations=False, res="i", dpi=600, mapflag='regional_ortho', mapfactor=2.75, geopolygons=mygeopolygons);

