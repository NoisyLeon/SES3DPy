import vmodel
import numpy as np
import domain
# outdir='./INPUT'
# Qmodel=vmodel.Qmodel( fmin=1.0/100., fmax=1/5.);
# # Qmodel=SES3DPy.Q_model( QArr=np.array([80.0]), fmin=1.0/1000., fmax=1/20.);
# Qmodel.Qdiscrete();
# 
# # D=np.array([1.684, 0.838, 1.357]);
# # tau_s=np.array([3.2, 17.692, 74.504]);
# # Qmodel.PlotQdiscrete( D=D, tau_s=tau_s );
# Qmodel.PlotQdiscrete();
# Qmodel.write(outdir=outdir)


smodel=vmodel.ses3d_model()
smodel.readh5model('../EAmodel.h5', modelname= 'EA_model_200_ses3d', maxdepth = 135)
smodel.smooth_horizontal(sigma=3, modelname='dvsv', filter_type='neighbour')
# smodel.read('./MODELS_3D')
# smodel.smooth_horizontal(sigma=10., modelname='dvsv', filter_type='gauss')
smodel.plot_slice(depth=3., modelname='dvsv')
# smodel.plot_threshold(val=3.5, min_val_plot=10., max_val_plot=40., modelname='dvsv')
# smodel.write('./MODELS_3D_test')
# smodel.read_block('./MODELS_3D')
# smodel.write_block('.')
# smodel.read_model('./MODELS_3D',  'dvp')
# smodel.write_model('.', 'dvp', verbose=True)