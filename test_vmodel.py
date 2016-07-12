import vmodel
import numpy as np

minlat=22.
maxlat=52.
minlon=85.
maxlon=133.
outdir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/MODELS/MODELS_3D'
smodel=vmodel.ses3d_model()
smodel.readh5model('../EAmodel.h5', minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon, modelname= 'EA_model_210_ses3d', maxdepth = 210.)
smodel.vsLimit(vsmin = 1.5)
smodel.smooth_horizontal(sigma=4, modelname='dvsv', filter_type='neighbour')
# smodel.read('./MODELS_3D')
# smodel.smooth_horizontal(sigma=10., modelname='dvsv', filter_type='gauss')
# smodel.plot_slice(depth=10., modelname='dvsv', min_val_plot=3.1, max_val_plot=3.8)
# smodel.plot_threshold(val=3.5, min_val_plot=10., max_val_plot=40., modelname='dvsv')
# smodel.write('./MODELS_3D_test')
# smodel.read_block('./MODELS_3D')
smodel.write(directory = outdir)
# smodel.read_model('./MODELS_3D',  'dvp')
# smodel.write_model('.', 'dvp', verbose=True)