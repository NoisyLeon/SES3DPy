import SES3DPy
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
stafile='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/station.lst_ses3d';

datadir='/pro/lustre/janus_scratch/life9360/EA_ses3d_working_dir/MODELS/MODELS'
mfname='dvsv'
rotationfile='/projects/life9360/software/ses3d_r07_b/TOOLS/rotation_parameters.txt'
setupfile='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT/setup'
recfile='/projects/life9360/software/ses3d_r07_b/INPUT/recfile_1'
sfield=SES3DPy.ses3d_fields(datadir, rotationfile, setupfile, recfile);

sfield.plot_depth_slice('vsh', 0, 1, 3, stations=False)
