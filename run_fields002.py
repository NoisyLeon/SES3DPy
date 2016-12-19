import fields
import numpy as np
import h5fields


#datadir='/lustre/janus_scratch/life9360/ses3d_working_2016_US/OUTPUT'
#setupfile='/lustre/janus_scratch/life9360/ses3d_working_2016_US/INPUT/setup'
datadir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/OUTPUT'
setupfile='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT/setup'

# # 
#sfield=fields.ses3d_fields(datadir, setupfile, field_type = "velocity_snapshot")
#mfname='vz'
#sfield.convert_to_hdf5('/lustre/janus_scratch/life9360/ses3d_h5fields/EA_10sec_20sec_10.h5', mfname, depth=0, diter=10)
# 
# dset=h5fields.h5fields('/lustre/janus_scratch/life9360/ses3d_h5fields/EA_10sec_20sec_10.h5')
# dset.zero_padding( outfname='/lustre/janus_scratch/life9360/ses3d_h5fields/EA_10sec_20sec_10_padding.h5', component='vz', minV=2.0, \
#                 evlo=129.0, evla=41.306, dt=0.05)
# dset.zero_padding( outfname='/lustre/janus_scratch/life9360/ses3d_h5fields/EA_10sec_20sec_10_padding_tt.h5', component='vz', minV=2.0, \
#                 evlo=129.0, evla=41.306, dt=0.05, iter0=0, iterf=30000, diter=5000)
dset2=h5fields.h5fields('/lustre/janus_scratch/life9360/ses3d_h5fields/EA_10sec_20sec_10_padding.h5')
dset2.convert_to_vts(outdir='/lustre/janus_scratch/life9360/ses3d_vts/EA_vts_10', component='vz')
# dset2.plot_depth_slice('vz', iteration=25000, vmin=-5e-7, vmax=5e-7)
# dset.plot_snapshots(component='vz', vmin=-3e-6, vmax=3e-6, outdir='/lustre/janus_scratch/life9360/h5test_snapshot' )
#dset.plot_snapshots_mp(component='vz', vmin=-5e-7, vmax=5e-7, outdir='/lustre/janus_scratch/life9360/h5test_snapshot_US')

