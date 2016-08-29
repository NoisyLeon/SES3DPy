import obspy
import symdata
import matplotlib.pyplot as plt
import numpy as np

prefname='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R/E000.98S43.pre'
f='SES.174S110.SAC'
f='./sac_major_second_seismogram/SES.98S43.SAC'
st=obspy.read(f)
tr=st[0]
tr1=symdata.ses3dtrace(tr.data, tr.stats)
# tr1.stats.sac.b=-0.0
tr1.aftan(piover4=-1., pmf=True, vmin=2.4, vmax=3.5, ffact=1. , tmin=6.0, tmax=15.0, phvelname=prefname)
tr1.plotftan()
plt.show()
# 
prefname='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R/E000.98S47.pre'
f='./sac_major_second_seismogram/SES.98S47.SAC'
st=obspy.read(f)
tr=st[0]
tr2=symdata.ses3dtrace(tr.data, tr.stats)
# tr1.stats.sac.b=-0.0
tr2.aftan(piover4=-1., pmf=True, vmin=2.4, vmax=3.5, ffact=1. , tmin=6.0, tmax=15.0, phvelname=prefname)
tr1.ftanparam.FTANcomp(tr2.ftanparam,  compflag=4)

plt.show()
