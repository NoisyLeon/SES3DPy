#!/usr/bin/env python
import symdata
import stations
import numpy as np
import obspy
import obspy.signal.filter
import obspy.signal.polarization
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
st = dbase.get_stream(staid='SES.98S45', SLst=SLst)

# stla, elve, stlo = dbase.waveforms['SES.98S45'].coordinates.values()
# print stlo, stla
# st[0].data=hilbert(st[0].data)
# st[1].data=hilbert(st[1].data)
st[2].data=hilbert(st[2].data)*-1.

stime=st[0].stats.starttime+643
etime=st[0].stats.starttime+720
rel= obspy.signal.polarization.polarization_analysis(stream=st, win_len=10., win_frac=0.1, frqlow=0.095, frqhigh=0.105, stime=stime, etime=etime, verbose=True,
            method='pm', var_noise=0.0)
# plt.plot(rel['timestamp'], rel['incidence'], 'bo')
# fig, ax=plt.subplots()
# plt.errorbar(rel['timestamp'], rel['incidence'], yerr=np.degrees(rel['incidence_error']))
# plt.subplots()
# ax.fill_betweenx(np.array([0, 80, 650, 670, facecolor='red', alpha=0.5)
ax.errorbar(rel['timestamp'], rel['azimuth'], yerr=np.degrees(rel['azimuth_error']), fmt='r')
# ax.plot(rel['timestamp'], rel['azimuth'], fmt='r')
# ax.fill_betweenx(np.array([0, 80]), 650, 670, facecolor='red', alpha=0.5)
# plt.show()
####################
st = dbase.get_stream(staid='SES.98S46', SLst=SLst)
# stla, elve, stlo = dbase.waveforms['SES.98S46'].coordinates.values()
# print stlo, stla
# st[0].data=hilbert(st[0].data)
# st[1].data=hilbert(st[1].data)
st[2].data=hilbert(st[2].data)*-1.

stime=st[0].stats.starttime+643
etime=st[0].stats.starttime+720
rel= obspy.signal.polarization.polarization_analysis(stream=st, win_len=10., win_frac=0.1, frqlow=0.08, frqhigh=0.125, stime=stime, etime=etime, verbose=True,
            method='pm', var_noise=0.0)
fig, ax=plt.subplots()
plt.errorbar(rel['timestamp'], rel['azimuth'], yerr=np.degrees(rel['azimuth_error']))

# 
# trE=st.select(channel='*E')[0]; trN=st.select(channel='*N')[0]; trZ=st.select(channel='*Z')[0]
# dataE = trE.data; dataN = trN.data; dataZ = trZ.data
# 
# plt.plot(dataZ, 'r-o')
# # plt.plot(hilbert(dataE), 'b-^')
# # plt.plot(hilbert(dataN), 'k-x')
# # plt.plot(hilbert(dataZ)*-1)
# plt.plot(dataE)
# plt.plot(dataN)
#
# st[2].plot(type='relative')
plt.show()