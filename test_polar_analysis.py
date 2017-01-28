#!/usr/bin/env python
import symdata
import stations
import numpy as np
import obspy
import obspy.signal.filter
import obspy.signal.polarization
import matplotlib.pyplot as plt
from scipy.fftpack import hilbert


def _aftan_gaussian_filter(alpha, omega0, ns, indata, omsArr):
    """Internal Gaussian filter function used for aftan
    """
    om2 = -(omsArr-omega0)*(omsArr-omega0)*alpha/omega0/omega0
    b=np.exp(om2)
    b[np.abs(om2)>=40.]=0.
    filterred_data=indata*b
    filterred_data[ns/2:]=0
    filterred_data[0]/=2
    filterred_data[ns/2-1]=filterred_data[ns/2-1].real+0.j
    return filterred_data


degree_sign= u'\N{DEGREE SIGN}'
datadir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/OUTPUT'
stafile='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT/recfile_1'
dbase=symdata.ses3dASDF('/lustre/janus_scratch/life9360/ASDF_data/ses3d_2016_10sec_3comp.h5')

SLst=stations.StaLst()
SLst.read(stafile)
evlo=129.0
evla=41.306
st = dbase.get_stream(staid='SES.98S46', SLst=SLst)
dt=st[0].stats.delta
# st.filter(type='bandpass', freqmin=0.095, freqmax=0.105)
# 
# ns=max(1<<(st[0].data.size-1).bit_length(), 2**12)  # different !!!
# sp0=np.fft.fft(st[0].data, ns); sp1 =np.fft.fft(st[1].data, ns); sp2=np.fft.fft(st[2].data, ns)
# domega = 2.*np.pi/ns/dt
# alpha=10.; T0=10.; omsArr=np.arange(ns)*domega
# 
# filterS0=_aftan_gaussian_filter(alpha=alpha, omega0=2*np.pi/T0, ns=ns, indata=sp0, omsArr=omsArr)
# filterS1=_aftan_gaussian_filter(alpha=alpha, omega0=2*np.pi/T0, ns=ns, indata=sp1, omsArr=omsArr)
# filterS2=_aftan_gaussian_filter(alpha=alpha, omega0=2*np.pi/T0, ns=ns, indata=sp2, omsArr=omsArr)
# 
# st[0].data=np.fft.ifft(filterS0, ns)
# st[1].data=np.fft.ifft(filterS1, ns)
# st[2].data=np.fft.ifft(filterS2, ns)


# st.plot(type='relative')
# st.decimate(10)
stla, elve, stlo = dbase.waveforms['SES.98S46'].coordinates.values()
dt=st[0].stats.delta
dist, az, baz_ev=obspy.geodetics.gps2dist_azimuth( stla, stlo, evla, evlo)
dist=dist/1000.


# st[0].data=hilbert(st[0].data)
# st[1].data=hilbert(st[1].data)
st[2].data=hilbert(st[2].data)*-1.

# st[1].data=st[2].data
# st[0].data=st[2].data


stime=st[0].stats.starttime+640
etime=st[0].stats.starttime+720
rel= obspy.signal.polarization.polarization_analysis(stream=st, win_len=10., win_frac=0.1, frqlow=0.02, frqhigh=0.2, stime=stime, etime=etime, verbose=True,
            method='pm', var_noise=0.5)
# plt.plot(rel['timestamp'], rel['incidence'], 'bo')
fig, ax=plt.subplots()
plt.errorbar(rel['timestamp'], rel['incidence'], yerr=np.degrees(rel['incidence_error']), lw=2, markersize=10)
ax.fill_betweenx( np.array([0, 80]), 660, 673, facecolor='red', alpha=0.5)
plt.xlim([645, 710])
plt.ylim([30, 50])
fig, ax=plt.subplots()
plt.errorbar(rel['timestamp'], rel['azimuth'], yerr=np.degrees(rel['azimuth_error']), lw=2, markersize=10)
ax.fill_betweenx( np.array([0, 80]), 660, 673, facecolor='red', alpha=0.5)
plt.xlim([645, 710])
plt.ylim([40, 80])
fig, ax=plt.subplots()
time=dt*np.arange(st[2].stats.npts)
plt.plot(time, st[0].data, lw=2, markersize=10)
ax.fill_betweenx( np.array([-2000, 2000]), 660, 673, facecolor='red', alpha=0.5)
plt.xlim([645, 710])
plt.ylim([-2000, 2000])

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