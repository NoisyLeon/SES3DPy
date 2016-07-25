import fields
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
stafile='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/station.lst_ses3d';

datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/MODELS/MODELS'
rotationfile='/projects/life9360/software/ses3d_r07_b/TOOLS/rotation_parameters.txt'
setupfile='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT/setup'
recfile='/projects/life9360/software/ses3d_r07_b/INPUT/recfile_1'
# datadir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/MODELS/MODELS'
# setupfile='/lustre/janus_scratch/life9360/SES3D_WorkingDir/INPUT/setup'

# datadir='/lustre/janus_scratch/life9360/bug_test/MODELS/MODELS'
# rotationfile='/lustre/janus_scratch/life9360/bug_test/TOOLS/rotation_parameters.txt'
# setupfile='/lustre/janus_scratch/life9360/bug_test/INPUT/setup'
# recfile='/lustre/janus_scratch/life9360/bug_test/INPUT/recfile_1'
sfield=fields.ses3d_fields(datadir, rotationfile, setupfile, recfile);
mfname='vsv'
sfield.plot_depth_slice(mfname, 3, 2500, 3500,  stations=False, res="i", mapflag='regional_merc',\
    mapfactor=2)
# sfield.plot_colat_slice('vsv', 42, 1000.0, 4500.0)
# sfield.plot_lat_depth_lon_slice('vsv', 35, depth=10., minlon=130., maxlon=135., valmin=1000.0, valmax=5000.0)


