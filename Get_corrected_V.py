import numpy as np
import numexpr as npr
from matplotlib.mlab import griddata
per_array=np.arange((50-10)/2+1)*2+10;
per_array=np.array([10]);
datadir='/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_TravelTime_MAP';
slowfname='slow_phase.map_HD';
Corrfname='NKNT_Corretion.lst_HD'
Vfname='PhaseV.map'
VfnameI='dPhaseV.map'
dlon=0.5;
dlat=0.5;
minlat=25.
maxlat=52.
minlon=90.
maxlon=143.
# slowfname='slow_azi_NKNT.phase.c.txt.HD.2.v2';
for per in per_array:
    Cdatadir=datadir+'/'+str(int(per))+'sec/';
    slowArr=np.loadtxt(Cdatadir+slowfname);
    lon1=slowArr[:,0];
    lat1=slowArr[:,1];
    Slow=slowArr[:,2];
    CorrArr=np.loadtxt(Cdatadir+Corrfname);
    lon2=CorrArr[:,0];
    lat2=CorrArr[:,1];
    CoLA=CorrArr[:,2];
    if npr.evaluate('sum(abs(lon1-lon2))')!=0:
        raise ValueError('Incompatible Amp Alplc data!')
    if npr.evaluate('sum(abs(lat1-lat2))')!=0:
        raise ValueError('Incompatible Amp Alplc data!')
    VArr=npr.evaluate('1./( sqrt(Slow**2-CoLA) )');
    VArr1=npr.evaluate('1./Slow');
    outArr=np.append(lon1,lat1);
    outArrI=np.append(outArr, VArr-VArr1)
    outArr=np.append(outArr,VArr);
    outArr=outArr.reshape((3,lon1.size));
    outArr=outArr.T;
    outArrI=outArrI.reshape((3,lon1.size));
    outArrI=outArrI.T;
    
    # 
    # Nlon=int((maxlon-minlon)/dlon)+1;
    # Nlat=int((maxlat-minlat)/dlat)+1;
    # mylon = np.linspace(minlon, maxlon, Nlon);
    # mylat = np.linspace(minlat, maxlat, Nlat);
    # SlowI = griddata(lon1, lat1, Slow, mylon, mylat);
    # CoLAI = griddata(lon2, lat2, CoLA, mylon, mylat);
    # VArrI=npr.evaluate('1./(sqrt(SlowI**2-CoLAI))');
    # outArrI=np.append(mylon,mylat);
    # outArrI=np.append(outArrI,VArrI);
    # outArrI=outArrI.reshape((3,mylon.size));
    # outArrI=outArrI.T;
    
    np.savetxt(Cdatadir+Vfname, outArr, fmt='%g');
    np.savetxt(Cdatadir+VfnameI, outArrI, fmt='%g');


    