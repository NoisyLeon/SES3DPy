import SES3DPy


datadir='/lustre/janus_scratch/life9360/bug_test/MODELS/MODELS'
rotationfile='/lustre/janus_scratch/life9360/bug_test/TOOLS/rotation_parameters.txt'
setupfile='/lustre/janus_scratch/life9360/bug_test/INPUT/setup'
recfile='/lustre/janus_scratch/life9360/bug_test/INPUT/recfile_1'

sfield=SES3DPy.ses3d_fields(datadir, rotationfile, setupfile, recfile);
mfname='vsv'
# sfield.plot_depth_slice('vsh', 3, 1000.0, 3000.0, stations=False)
sfield.CheckVelocityLimit(mfname)
mfname='vsh'
sfield.CheckVelocityLimit(mfname)
mfname='vp'
# sfield.plot_depth_slice('vsh', 3, 1000.0, 3000.0, stations=False)
sfield.CheckVelocityLimit(mfname)
mfname='rho'
# sfield.plot_depth_slice('vsh', 3, 1000.0, 3000.0, stations=False)
sfield.CheckVelocityLimit(mfname)

