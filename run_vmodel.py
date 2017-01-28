import vmodel
import numpy as np
import GeoPolygon

#############################################################
# study region
#############################################################
# minlat=22.
# maxlat=52.
# minlon=85.
# maxlon=133.

minlat=24.
maxlat=50.
minlon=-125.0
maxlon=-66.
############################################################## 
outdir='/lustre/janus_scratch/life9360/ses3d_working_2016_US/MODELS/MODELS_3D'
smodel=vmodel.ses3d_model()
# smodel.readh5model('../USmodel.h5', minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon, groupname= 'US_model_410km_ses3d', maxdepth = 210.)
# smodel.vsLimit(vsmin = 2.0)
# smodel.smooth_horizontal(sigma=4, filter_type='neighbour')
# smodel.convert_to_vts(outdir='../USmodel_vtk', modelname='dvsv')
# smodel.vsLimit(vsmin = 2.0)
# smodel.smooth_horizontal(sigma=4, filter_type='neighbour')

# 
# basins=GeoPolygon.GeoPolygonLst()
# basins.ReadGeoPolygonLst('basin1')
# 
smodel.read(directory = outdir)
# smodel.projection='lambert'
# smodel.plot_slice(depth=3., modelname='dvsv', min_val_plot=2.6, max_val_plot=3.6)
# smodel.plot_slice(depth=20., modelname='dvsv', min_val_plot=3.2, max_val_plot=4.0, geopolygons=basins)
# smodel.plot_slice(depth=10., modelname='dvsv', min_val_plot=3.0, max_val_plot=3.8, geopolygons=basins)
# smodel.plot_slice(depth=10., modelname='dvp')
# smodel.plot_threshold(val=3.5, min_val_plot=10., max_val_plot=40., modelname='dvsv')
# smodel.write('./MODELS_3D_for_paper')
# smodel.read_block('./MODELS_3D')
# smodel.write(directory = outdir)
# smodel.read_model('./MODELS_3D',  'dvp')
# smodel.write_model('.', 'dvp', verbose=True)
