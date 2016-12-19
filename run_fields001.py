import fields
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import numpy as np
import GeoPolygon
#datadir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/MODELS/MODELS'
datadir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/OUTPUT'
# rotationfile='/projects/life9360/software/ses3d_r07_b/TOOLS/rotation_parameters.txt'
setupfile='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT/setup'

basins=GeoPolygon.GeoPolygonLst()
basins.ReadGeoPolygonLst('basin1')

#setupfile='/home/lili/code/INPUT/setup'
#datadir='/home/lili/code/snapshots'
sfield=fields.ses3d_fields(datadir, setupfile, field_type = "velocity_snapshot");
mfname='vz'
# sfield.plot_snapshots_all6_mp( mfname, depth=0, valmin=-2e-7, valmax=2e-7, outdir='/lustre/janus_scratch/life9360/EA_snapshots_2016_all6', fprx='wavefield',
#         iter0=30000, iterf=32000, diter=200, stations=False, res="i", proj='lambert', dpi=300, zoomin=2, geopolygons=basins,
#             evlo=129.0, evla=41.306, vpadding=2.0, dt=0.05)
sfield.plot_snapshots_mp( mfname, depth=0, valmin=-2e-7, valmax=2e-7, outdir='/lustre/janus_scratch/life9360/EA_snapshots_2016_for_paper', fprx='wavefield',
        iter0=18000, iterf=24000, diter=200, stations=False, res="i", proj='lambert', dpi=300, zoomin=2, geopolygons=basins,
            evlo=129.0, evla=41.306, vpadding=2.0, dt=0.05)
# sfield.check_model(mfname)
# sfield.plot_depth_slice(mfname, 3., 2500, 3500,  stations=False, res="l", mapflag='global',\
#     zoomin=2)
evlo=129.0
# evlo=129.029
evla=41.306
# sfield.plot_depth_padding_slice(mfname, 0., valmin=-1e-7, valmax=1e-7, dt=0.05, vpadding=2.7, 
#        evlo=evlo, evla=evla, stations=False, res="l", proj='lambert', iteration=15000, zoomin=2.5)
# sfield.plot_depth_padding_all6_slice(mfname, 0., valmin=-1e-7, valmax=1e-7, dt=0.05, vpadding=2.7, 
#        evlo=evlo, evla=evla, stations=False, res="l", proj='lambert', iteration=2000, zoomin=2.5)
# sfield.plot_depth_slice(mfname, 0., valmin=-1e-7, valmax=1e-7, evlo=evlo, evla=evla, stations=False, res="l", proj='regional_ortho', iteration=15000, zoomin=2.5)
# sfield.plot_lat_depth_lon_slice(mfname, lat=41.,depth=0., minlon =90., maxlon = 120., valmin=-6e-8, valmax=6e-8, iteration=4000)
# sfield.plot_colat_slice('vsv', 42, 1000.0, 4500.0)
# sfield.plot_lat_depth_lon_slice('vsv', 35, depth=10., minlon=130., maxlon=135., valmin=1000.0, valmax=5000.0)
