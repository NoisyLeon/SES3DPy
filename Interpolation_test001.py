import numpy as np
import matplotlib.pyplot as plt
import GeoPoint
import scipy.interpolate
infname='/projects/life9360/EA_MODEL/SEA_MODEL/122.5_33.5_mod'
ak135fname='/projects/life9360/code/ses3dPy/ak135.mod'


depth=np.array([10.,100.,200.])
dz=np.array([0.5,1.,2.])
layerarr1=GeoPoint.GenerateDepthArr(depth, dz)
inArr=np.loadtxt(infname);
depth=inArr[:,0];
Vp=inArr[:,2];
depinter=layerarr1
Vpinter=np.interp(depinter, depth, Vp);


inak135Arr=np.loadtxt(ak135fname);
depthak135=inak135Arr[:,0];
Vpak135=inak135Arr[:,2];

Vpak135_app=Vpak135[(depthak135==410.)]
depthak135_app=depthak135[(depthak135==410.)]

depinter2=200.+np.arange((410.-210.)/10.+1)*10.+10.
Vpak135_app=np.append(Vp[-1], Vpak135_app[-2])
depthak135_app=np.append(depth[-1], 410.)
# fspline= scipy.interpolate.UnivariateSpline(depthak135_app, Vpak135_app, k=1)
# 
# Vpinter2=fspline(depinter2);
Vpinter2=np.interp(depinter2, depthak135_app, Vpak135_app);

Vpinter=np.append(Vpinter,Vpinter2);
depinter=np.append(depinter,depinter2);
depinter=np.append(depinter,depthak135[(depthak135<500.)*(depthak135>410.)]);
Vpinter=np.append(Vpinter,Vpak135[(depthak135<500.)*(depthak135>410.)]);



# sVp=scipy.interpolate.piecewise_polynomial_interpolate(depinter, Vpinter, depinter)
ax=plt.subplot(1,1,1)
ax.plot(depinter, Vpinter, '^b');
ax.plot(depth, Vp, 'xr');
ax.plot(depthak135[(depthak135<500.)], Vpak135[(depthak135<500.)], '.-g');
# ax.plot(depinter, sVp, 'xr');
# ax=plt.subplot(2,1,2)
# ax.plot(depth, abs(Vp-Vpinter), 'or');
plt.show()