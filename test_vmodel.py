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
# dx=smodel.read('/projects/life9360/code/ses3d_janus/MODELS_c/MODELS_3D', 'dvsv')
smodel.read('/home/lili/code/ses3d_r07_b/MODELS/MODELS_3D',  'dvsv')
smodel.write('.', 'dvsv_new', verbose=True)