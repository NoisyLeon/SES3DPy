#!/usr/bin/env python
import symdata
import stations
import numpy as np
import obspy
import obspy.signal.filter
import matplotlib.pyplot as plt
from scipy.fftpack import hilbert

degree_sign= u'\N{DEGREE SIGN}'
datadir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/OUTPUT'
stafile='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT/recfile_1'
dbase=symdata.ses3dASDF('/lustre/janus_scratch/life9360/ASDF_data/ses3d_2016_10sec_3comp.h5')

SLst=stations.StaLst()
SLst.read(stafile)
evlo=129.0
evla=41.306
st = dbase.get_stream(staid='SES.98S47', SLst=SLst)
st.decimate(10)
stla, elve, stlo = dbase.waveforms['SES.98S47'].coordinates.values()
dt=st[0].stats.delta
dist, az, baz_ev=obspy.geodetics.gps2dist_azimuth( stla, stlo, evla, evlo)
dist=dist/1000.
bazArr=baz_ev+np.arange(50)-25. ###
# 
# tmin=640; tmax=710; twin=2; tlength=20;
vmin = 2.; vmax = 5.
tmin=dist/vmax; tmax=dist/vmin; twin=1; tlength=20

maxbaz=np.zeros(int((tmax-tmin)/twin))
N0=np.floor((np.arange(int((tmax-tmin)/twin))*twin+tmin)/dt)
Nt=np.ceil((np.arange(int((tmax-tmin)/twin))*twin+tmin+tlength)/dt)
Tmed=(Nt+N0)*dt/2
NminArr=np.ones(int((tmax-tmin)/twin))*999999
for baz in bazArr:
    print baz
    st2=st.copy()
    st2.rotate(method='NE->RT', back_azimuth=baz)
    trT=st2.select(channel='*T')[0]
    trR=st2.select(channel='*R')[0]
    envT = obspy.signal.filter.envelope(trT.data)
    envR = obspy.signal.filter.envelope(trR.data)
    for i in xrange(Nt.size):
        n0=N0[i]; nt=Nt[i]
        idata_enve=envT[n0:nt]
        if NminArr[i]>idata_enve.sum()/(nt-n0):
            NminArr[i]=idata_enve.sum()/(nt-n0)
            maxbaz[i]=baz
# 
# dist, az, baz=obspy.geodetics.gps2dist_azimuth( 34, 110, evla, evlo)    
st3=st.copy()
st3.rotate(method='NE->RT', back_azimuth=baz_ev)
ydata=st3[1].data
time=np.arange(st3[1].stats.npts)*dt
# 
fig, ax1 = plt.subplots()
plt.plot(time,ydata, 'k-', lw=5)
ax1.set_ylabel('Amplitude (nm)', color='k', fontsize=20)
ax1.tick_params(labelsize=20)
ax2 = ax1.twinx()
# ax2.plot(Tmed, maxbaz+180, 'b.', markersize=20)
ax2.plot(Tmed, maxbaz, 'b.', markersize=20)
ax2.plot(Tmed, np.ones(Tmed.size)*248, 'r-', lw=5)
ax2.plot(Tmed, np.ones(Tmed.size)*231, 'g-', lw=5)
plt.xlim(550, 750)
ax2.set_ylabel('Propagation Angle ('+degree_sign+')', color='b', fontsize=20)
ax2.set_yticks(np.arange(225, 260, 5))
ax2.set_ylim(225, 255)
# # ax2.set_yticks(np.arange(45, 71, 6))
# # ax2.set_ylim(225, 250)
# ax2.tick_params(labelsize=20)
# ax1.set_xlabel('Time (sec)', fontsize=20)
# for tl in ax2.get_yticklabels():
#     tl.set_color('b')
plt.show()
# 
trZ=st3.select(channel='*Z')[0]
trR=st3.select(channel='*R')[0]
ratio=hilbert(trR.data).max()/trZ.data.max()
plt.plot(hilbert(trR.data)/ratio)
plt.plot(trZ.data)
plt.show()
# 
# dbase.readtxt(datadir=datadir, stafile=stafile, channel='all', verbose=True, VminPadding=2.7)
# dbase.readtxt(datadir=datadir, stafile=stafile, channel='all', verbose=True, VminPadding=2.0, factor=10)

# evlo=129.0
# evla=41.306
# try:
#     del dbase.events
# except:
#     pass
# dbase.AddEvent(evlo, evla, evdp=1.0)

# # # dset.pre_php(outdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016')
# # dbase.pre_php(outdir='/lustre/janus_scratch/life9360/ses3d_2016_US_phv/PRE_PHP')
# 
# # # # 
# inftan=symdata.InputFtanParam()
# inftan.ffact=1.
# inftan.vmin=2.1
# inftan.pmf=True
# # # 
# 
# # # 
# # # # try:
# # # #     del dbase.auxiliary_data.DISPpmf2interp
# # # # except:
# # # #     pass
# # # # # 
# # # # try:
# # # #     del dbase.auxiliary_data.FieldDISPpmf2interp
# # # # except:
# # # #     pass
# # # 
# try:
#     del dbase.auxiliary_data.DISPbasic1
# except:
#     pass
# try:
#     del dbase.auxiliary_data.DISPbasic2
# except:
#     pass
# try:
#     del dbase.auxiliary_data.DISPpmf1
# except:
#     pass
# try:
#     del dbase.auxiliary_data.DISPpmf2
# except:
#     pass
# # # # 
# dbase.aftanMP(outdir='/lustre/janus_scratch/life9360/ses3d_working_2016_US/DISP', inftan=inftan, basic2=True,
#             pmf1=True, pmf2=True, prephdir='/lustre/janus_scratch/life9360/ses3d_2016_US_phv/PRE_PHP_R', tb=0.0)
# # # # dbase.aftanMP(outdir='/lustre/janus_scratch/life9360/DISP_2015', inftan=inftan, basic2=True,
# # # #             pmf1=True, pmf2=True, prephdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2015_R', tb=0.0)
# # # 
# # # 
# # # # dbase.aftan(inftan=inftan, basic2=True,
# # # #             pmf1=True, pmf2=True, prephdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R')
# # # # dbase.aftan(tb=-11.9563813705, outdir='/lustre/janus_scratch/life9360/LFMembrane_SH_0.1_20/DISP', inftan=inftan, basic2=True,
# # # #             pmf1=True, pmf2=True)
# # # 
# try:
#     del dbase.auxiliary_data.DISPpmf2interp
# except:
#     pass
# dbase.interpDisp(data_type='DISPpmf2')
# 
# try:
#     del dbase.auxiliary_data.FieldDISPpmf2interp
# except:
#     pass
# # # dbase.get_field(outdir='./stf_10sec', fieldtype='amp', data_type='DISPpmf2')
# # # dbase.get_field(outdir='./stf_10sec', fieldtype='Vgr',  data_type='DISPpmf2')
# # # dbase.get_field(outdir='./stf_10sec', fieldtype='Vph',  data_type='DISPpmf2')
# dbase.get_all_field(outdir='/lustre/janus_scratch/life9360/ses3d_field_working/stf_100_10sec_US', data_type='DISPpmf2')
# # dbase.get_all_field(outdir='./stf_10_20sec', data_type='DISPpmf2')
# 
# dbase.get_field(outdir='./stf_10_20sec', data_type='DISPpmf2', fieldtype='ms')
