import numpy as np
import numexpr as npr
per_array=np.arange((50-10)/2+1)*2+10;
datadir='/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_TravelTime_MAP';
Ampfname='NKNT_Amplitude.lst';
Corrfname='NKNT_Corretion.lst'
# slowfname='slow_azi_NKNT.phase.c.txt.HD.2.v2';
dx=0.5;
for per in per_array:
    Cdatadir=datadir+'/'+str(int(per))+'sec/';
    inAmpArr=np.loadtxt(Cdatadir+Ampfname);
    lon1=inAmpArr[:,0];
    lat1=inAmpArr[:,1];
    Amp=inAmpArr[:,2];
    inAlplcArr=np.loadtxt(Cdatadir+Ampfname+'_lplc');
    lon2=inAlplcArr[:,0];
    lat2=inAlplcArr[:,1];
    Alplc=inAlplcArr[:,2];
    if npr.evaluate('sum(abs(lon1-lon2))')!=0:
        raise ValueError('Incompatible Amp Alplc data!')
    if npr.evaluate('sum(abs(lat1-lat2))')!=0:
        raise ValueError('Incompatible Amp Alplc data!')
    omega=2*np.pi/float(per);
    omega2=omega**2;
    CorrArr=npr.evaluate('Alplc/Amp/omega2');
    outlon=lon1[(Alplc!=0)*( Alplc <1)*(Alplc > -1)];
    outlat=lat1[(Alplc!=0)*( Alplc <1)*(Alplc > -1)];
    CorrArr=CorrArr[(Alplc!=0)*( Alplc <1)*(Alplc > -1)];
    outArr=np.append(outlon,outlat);
    outArr=np.append(outArr,CorrArr);
    outArr=outArr.reshape((3,outlon.size));
    outArr=outArr.T;
    np.savetxt(Cdatadir+Corrfname, outArr, fmt='%g');


    