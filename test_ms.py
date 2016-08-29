import obspy
import symdata
import matplotlib.pyplot as plt
import numpy as np


prefname='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R/E000.98S43.pre'
# f='SES.174S110.SAC'
f='./SES.98S43.SAC'
f='/projects/life9360/instaseis_seismogram/S001_INSTASEIS.LXZ.SAC'
st=obspy.read(f)
tr=st[0]
tr.data=tr.data*1e9
tr1=symdata.ses3dtrace(tr.data, tr.stats)
# tr1.stats.sac.b=-0.0
tr1.aftan()
# for f in np.arange(100)*0.05+0.1:
print tr1.get_ms(period=10., wfactor=1.)