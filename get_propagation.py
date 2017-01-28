import symdata
import stations
import obspy
import obspy.signal.filter
import numpy as np
import matplotlib.pyplot as plt
stafile='/work3/leon/ASDF_data/recfile_1'
dset = symdata.ses3dASDF('/work3/leon/ASDF_data/ses3d_2016_10sec_3comp.h5')

degree_sign= u'\N{DEGREE SIGN}'
evlo=129.029
evla=41.306

st=dset.waveforms['SES.98S46'].ses3d_raw
for tr in st:
    tr.stats.sac={}
    tr.stats.sac['evlo']=129.029; tr.stats.sac['evla']=41.306
    tr.stats.sac['stlo']=110; tr.stats.sac['stla']=34
# st.decimate(10)
st3=obspy.Stream()
# bazArr=220.+np.arange(50)
bazArr=48.+np.arange(50)
dt=st[0].stats.delta
tmin=640; tmax=710; twin=2; tlength=20;
maxbaz=np.zeros(int((tmax-tmin)/twin)); N0=(np.arange(int((tmax-tmin)/twin))*twin+tmin)/dt
Nt=(np.arange(int((tmax-tmin)/twin))*twin+tmin+tlength)/dt; Tmed=(Nt+N0)*dt/2
NminArr=np.ones(int((tmax-tmin)/twin))*999999
for baz in bazArr:
    st2=st.copy()
    st2.rotate(method='NE->RT', back_azimuth=baz)
    trT=st2[0]
    data_envelope = obspy.signal.filter.envelope(trT.data)
    for i in xrange(Nt.size):
        n0=N0[i]; nt=Nt[i]
        idata_enve=data_envelope[n0:nt]
        if NminArr[i]>idata_enve.sum()/(nt-n0):
            NminArr[i]=idata_enve.sum()/(nt-n0)
            maxbaz[i]=baz

dist, az, baz=obspy.geodetics.gps2dist_azimuth( 34, 110, evla, evlo)    
st3=st.copy()
st3.rotate(method='NE->RT', back_azimuth=baz)
ydata=st3[1].data/16649.0*1659.53
time=np.arange(st3[1].stats.npts)*dt

fig, ax1 = plt.subplots()
plt.plot(time,ydata, 'k-', lw=5)
ax1.set_ylabel('Amplitude (nm)', color='k', fontsize=20)
ax1.tick_params(labelsize=20)
ax2 = ax1.twinx()
ax2.plot(Tmed, maxbaz+180, 'b.', markersize=20)
ax2.plot(Tmed, np.ones(Tmed.size)*248, 'r-', lw=5)
ax2.plot(Tmed, np.ones(Tmed.size)*231, 'g-', lw=5)
plt.xlim(550, 750)
ax2.set_ylabel('Propagation Angle ('+degree_sign+')', color='b', fontsize=20)
ax2.set_yticks(np.arange(225, 260, 5))
ax2.set_ylim(225, 255)
# ax2.set_yticks(np.arange(45, 71, 6))
# ax2.set_ylim(225, 250)
ax2.tick_params(labelsize=20)
ax1.set_xlabel('Time (sec)', fontsize=20)
for tl in ax2.get_yticklabels():
    tl.set_color('b')
plt.show()
    

# print az, baz
# 
# for i in xrange(11):
#     dbaz=i-30
#     st2=st.copy()
#     st2.rotate(method='NE->RT', back_azimuth=baz+dbaz)
#     tr=st2[0]
#     tr.stats.station+=str(i)
#     data_envelope = obspy.signal.filter.envelope(st2[0].data)
#     tr.data=data_envelope
#     st3.append(tr)
# st3.append(st[2])
# startT=st[0].stats.starttime