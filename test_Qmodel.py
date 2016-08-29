
import Qmodel

outdir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT'
#########################
# Qmodel

Qmodel=Qmodel.Qmodel( fmin=1.0/100., fmax=1/10.)
Qmodel.Qdiscrete()

# D=np.array([1.684, 0.838, 1.357]);
# tau_s=np.array([3.2, 17.692, 74.504]);
# Qmodel.PlotQdiscrete( D=D, tau_s=tau_s );
Qmodel.PlotQdiscrete()
# Qmodel.write(outdir=outdir)
#########################