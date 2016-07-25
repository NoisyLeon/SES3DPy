import fields
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np

# datadir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/MODELS/MODELS'
datadir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/OUTPUT'
# rotationfile='/projects/life9360/software/ses3d_r07_b/TOOLS/rotation_parameters.txt'
setupfile='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT/setup'

setupfile='/home/lili/code/INPUT/setup'
datadir='/home/lili/code/snapshots'
sfield=fields.ses3d_fields(datadir, setupfile, field_type = "velocity_snapshot");
mfname='vz'
# sfield.plot_snapshots_mp( mfname, depth=0, valmin=-6e-6, valmax=6e-6, outdir='/lustre/janus_scratch/life9360/EA_snapshots_2016',
#             fprx='wavefield',iter0=3000, iterf=4000, diter=200, stations=False, res="i", proj='global', dpi=300, zoomin=2, geopolygons=None, evlo=129.0, evla=41.306)
# sfield.check_model(mfname)
# sfield.plot_depth_slice(mfname, 3., 2500, 3500,  stations=False, res="l", mapflag='global',\
#     zoomin=2)
evlo=129.0
# evlo=129.029
evla=41.306
sfield.plot_depth_padding_slice(mfname, 0., valmin=-3e-6, valmax=3e-6, dt=0.05, vpadding=2.7, 
        evlo=evlo, evla=evla, stations=False, res="l", proj='regional_ortho', iteration=7200, zoomin=2.5)
# sfield.plot_lat_depth_lon_slice(mfname, lat=41.,depth=0., minlon =90., maxlon = 120., valmin=-6e-8, valmax=6e-8, iteration=4000)
# sfield.plot_colat_slice('vsv', 42, 1000.0, 4500.0)
# sfield.plot_lat_depth_lon_slice('vsv', 35, depth=10., minlon=130., maxlon=135., valmin=1000.0, valmax=5000.0)
