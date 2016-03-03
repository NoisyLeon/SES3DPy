import numpy as np

mfname='/projects/life9360/software/fk/ak135_ori';
outmfname='/projects/life9360/software/fk/ak135';
InArr=np.loadtxt(mfname);
dep=InArr[:,0];
Vp=InArr[:,1];
Vs=InArr[:,2];
L=dep.size;
dep0=dep[:L-1];
dep1=dep[1:];
thick=dep1-dep0;

Vp0=Vp[:L-1];
Vp1=Vp[1:];
Vpavg=(Vp0+Vp1)/2;

Vs0=Vs[:L-1];
Vs1=Vs[1:];
Vsavg=(Vs0+Vs1)/2;

OutArr=np.append(thick, Vsavg);
OutArr=np.append(OutArr, Vpavg);
OutArr=OutArr.reshape((3,L-1));
OutArr=OutArr.T;
np.savetxt(outmfname, OutArr, fmt='%g');