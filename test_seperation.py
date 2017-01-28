#!/usr/bin/env python
import symdata
import stations
import numpy as np
import obspy
import obspy.signal.filter
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
st = dbase.get_stream(staid='SES.98S47', SLst=SLst)
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


st.plot(type='relative')
st.decimate(10)
stla, elve, stlo = dbase.waveforms['SES.98S47'].coordinates.values()
dt=st[0].stats.delta
dist, az, baz_ev=obspy.geodetics.gps2dist_azimuth( stla, stlo, evla, evlo)
dist=dist/1000.

vmin = 2.; vmax = 5.
tmin=dist/vmax; tmax=dist/vmin; twin=1; tlength=20

tmin=640; tmax=700
n0 = int(tmin/dt); nt = int(tmax/dt)
trE=st.select(channel='*E')[0]; trN=st.select(channel='*N')[0]; trZ=st.select(channel='*Z')[0]
dataE = trE.data[n0:nt]; dataN = trN.data[n0:nt]; dataZ = trZ.data[n0:nt]
dataE = trE.data; dataN = trN.data; dataZ = trZ.data

plt.plot(hilbert(dataZ)*-1)
plt.plot(dataE)
# plt.scatter(dataE, dataN, c=n0+np.arange(nt-n0), cmap='jet')
# plt.subplots()
# plt.scatter( n0+np.arange(nt-n0), dataZ, c=n0+np.arange(nt-n0), cmap='jet')

plt.show()