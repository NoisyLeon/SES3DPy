import obspy
import numpy as np
import obspy.core.util.geodetics as obsGeo
import matplotlib.pyplot as plt
lat=2*100
lonLst1=1510+np.arange(30)*10;
lonLst2=1490-np.arange(30)*10;
evlo=15.;
evla=2.;
datadir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT_SAC_TEST_PML_10sec_001'
dist, az, baz=obsGeo.gps2DistAzimuth(evla, evlo, 2., 18.0 )
Lmax=dist/1000.
ampArr=np.array([])
L=np.array([])
rmsArr=np.array([])
Lambda=40.;

for i in np.arange(30):
    lon1=lonLst1[i];
    lon2=lonLst2[i];
    infname1=datadir+'/LF.LF'+str(lat)+'S'+str(lon1)+'..BXZ.SAC';
    infname2=datadir+'/LF.LF'+str(lat)+'S'+str(lon2)+'..BXZ.SAC';
    tr1=obspy.read(infname1)[0];
    tr2=obspy.read(infname2)[0];
    A1max=(np.abs(tr1.data)).max();
    A2max=(np.abs(tr2.data)).max();
    dist, az, baz=obsGeo.gps2DistAzimuth(evla, evlo, float(lat)/100, float(lon2)/100 )
    delA=(A2max-A1max)/A2max;
    ampArr=np.append(ampArr,delA);
    L=np.append(L, (Lmax-dist/1000.)/Lambda);
    diffArr=np.abs(tr1.data-tr2.data);
    endT=dist/2000.;
    dt=tr1.stats.delta;
    endIndex=int(endT/dt);
    diffArr=diffArr[:endIndex];
    rms=np.sqrt( (diffArr**2).sum()/diffArr.size )
    rmsArr=np.append(rmsArr, rms)
    
degree_sign= u'\N{DEGREE SIGN}'
ax=plt.subplot(111);
# ax.semilogy(L, ampArr, 'o', markersize=20);
ax.semilogy(L, rmsArr, 'o', markersize=20);
plt.xlabel('L/$\lambda$', fontsize=20)
# plt.ylabel('$\Delta$A/A', fontsize=20)
plt.ylabel('RMS', fontsize=20)
plt.xlim(-0.5, 8.5)
plt.ylim(10e-5, 100)
plt.title('RMS for major arrival',fontsize=30)
# plt.ylim(10e-8, 5.)
plt.show()

    

