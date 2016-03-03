import obspy
import matplotlib.pyplot as plt
import numpy as np
f1='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT_SAC_TEST_PML_10sec_001/\
LF.LF200S1800..BXZ.SAC';
f2='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT_SAC_TEST_PML_10sec_001/\
LF.LF200S1200..BXZ.SAC';
tr1=obspy.read(f1)[0];
tr2=obspy.read(f2)[0];
ax=plt.subplot(111);
time=np.arange(tr1.stats.npts)*tr1.stats.delta
line1,=ax.plot(time, tr1.data, 'r.-', lw=2 );
line2,=ax.plot(time, tr2.data, 'b--', lw=2 );
ax.legend([line1,line2], ['lat: 2, lon: 18', 'lat: 2, lon: 12'])

plt.show()