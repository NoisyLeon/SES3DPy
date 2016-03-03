import obspy
import ses3d_noisepy as npy
# import ses3d_noisepy as snpy
import matplotlib.pyplot as plt
import numpy as np


f1='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT_SAC_TEST_PML_10sec_001/LF.LF370S1100..BXZ.SAC'
f1='/projects/life9360/instaseis_seismogram/S001_INSTASEIS.LXZ.SAC'
prefname='/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_DISP/PREPHASE_R/NKNT.910S420.pre'
st=obspy.read(f1)
tr=st[0]
tr1=npy.noisetrace(tr.data, tr.stats)
# tr1.Makesym()
tr1.aftan(piover4=-1., pmf=True, vmin=2.5, tmin=5.0, tmax=70.0, phvelname=prefname)
tr1.getSNRRussell();
tr1.plotftan()
plt.show()


