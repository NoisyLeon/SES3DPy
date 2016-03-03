import SES3DPy
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
stafile='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/station.lst_ses3d';
# datadir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/MODELS/MODELS_3D';
datadir='/projects/life9360/software/ses3d_r07_b/MODELS/MODELS'
# sfield=SES3DPy.ses3d_fields(datadir, rotationfile, setupfile, recfile);

# datadir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT_ak135_002'
# rotationfile='/lustre/janus_scratch/life9360/SES3D_WorkingDir/TOOLS/rotation_parameters.txt'
# setupfile='/lustre/janus_scratch/life9360/SES3D_WorkingDir/INPUT/setup'
# recfile='/lustre/janus_scratch/life9360/SES3D_WorkingDir/INPUT/recfile_1'
# outdir='/lustre/janus_scratch/life9360/snapshots_ortho_parallel_test';

datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT'
rotationfile='/lustre/janus_scratch/life9360/SES3D_WorkingDir/TOOLS/rotation_parameters.txt'
setupfile='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT/setup'
recfile='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/INPUT/recfile_1'
outdir='/lustre/janus_scratch/life9360/snapshots_ortho_parallel_EA_001_100_pdf';

evlo=129.029;
evla=41.306;
sfield=SES3DPy.ses3d_fields(datadir, rotationfile, setupfile, recfile, field_type="velocity_snapshot");
# sfield.plot_colat_slice(component='vz', colat=49, valmin=-3e-8, valmax=3e-8, iteration=4600, verbose=True)
sfield.MakeAnimationParallel(evlo=evlo, evla=evla, component='vz', depth=0, valmin=-5e-7, valmax=5e-7, outdir=outdir,\
        iter0=12100, iterf=20100, dsnap=100, stations=False, res="i", mapflag='regional_ortho', mapfactor=4)


