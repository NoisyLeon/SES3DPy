
"""
This is a sub-module of noisepy.
Classes and functions for geographycal points analysis and plotting.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from matplotlib.mlab import griddata
import matplotlib.cm
from matplotlib.ticker import FuncFormatter
import numexpr as npr
import glob
from functools import partial
import multiprocessing as mp
import obspy.core.util.geodetics as obsGeo
import math
from geopy.distance import great_circle
from netCDF4 import Dataset
import pycpt
import shutil

# setup Lambert Conformal basemap.
# set resolution=None to skip processing of boundary datasets.
# m = Basemap(width=12000000,height=9000000,projection='lcc',
#             resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
# m.etopo()

class GeoPoint(object):
    """
    A class for a geographycal point analysis
    ---------------------------------------------------------------------
    Parameters:
    name - lon_lat
    lon, lat
    depthP - depth profile (np.array)
    depthPfname - 
    DispGr - Group V dispersion Curve (np.array)
    GrDispfname - 
    DispPh - Phase V dispersion Curve (np.array)
    PhDispfname - 
    """
    def __init__(self, name='',lon=None, lat=None, depthP=np.array([]), depthPfname='', DispGr=np.array([]), GrDispfname='',\
        DispPh=np.array([]), PhDispfname=''):
        self.name=name
        self.lon=lon
        self.lat=lat
        self.depthP=depthP
        self.depthPfname=depthPfname
        self.DispGr=DispGr
        self.GrDispfname=GrDispfname
        self.DispPh=DispPh
        self.PhDispfname=PhDispfname
        self.user0=None;
        self.user1=None;
        self.user2=None;
        self.user3=None;
    
    def SetName(self,name=''):
        if name=='':
            self.name='%g'%(self.lon) + '_%g' %(self.lat)
        else:
            self.name=name
        return
    
    def SetVProfileFname(self, prefix='MC.', suffix='.acc.average.mod.q'):
        self.SetName();
        self.depthPfname=prefix+self.name+suffix
        return
    
    def SetGrDispfname(self, prefix='MC.', suffix='.acc.average.g.disp'):
        self.GrDispfname=prefix+self.name+suffix
        return
    
    def SetPhDispfname(self, prefix='MC.', suffix='.acc.average.p.disp'):
        self.PhDispfname=prefix+self.name+suffix
        return
    
    def SetAllfname(self):
        self.SetVProfileFname()
        self.SetGrDispfname()
        self.SetPhDispfname()
        return
      
    def LoadVProfile(self, datadir='', dirPFX='', dirSFX='', depthPfname=''):
        if depthPfname!='':
            infname=depthPfname
        elif datadir=='':
            infname='./'+dirPFX+self.name+dirSFX+'/'+self.depthPfname
        elif dirPFX==None or dirSFX==None:
            infname=datadir+'/'+self.depthPfname
        else:
            infname=datadir+'/'+dirPFX+self.name+dirSFX+'/'+self.depthPfname
        if os.path.isfile(infname):
            self.depthP=np.loadtxt(infname)
        else:
            # print infname
            print 'Warning: No Depth Profile File for:'+str(self.lon)+' '+str(self.lat)
        return
    
    def SaveVProfile(self, outdir='', dirPFX='', dirSFX='', depthPfname=''):
        if depthPfname!='':
            outfname=depthPfname
        elif outdir=='':
            outfname='./'+dirPFX+self.name+dirSFX+'/'+self.depthPfname
        elif dirPFX==None or dirSFX==None:
            outfname=outdir+'/'+self.depthPfname
        else:
            outfname=outdir+'/'+dirPFX+self.name+dirSFX+'/'+self.depthPfname
        np.savetxt(outfname, self.depthP, fmt='%g');
        return;
    
    def ReplaceVProfile(self, outdir='', dirPFX='', dirSFX='', depthPfname=''):
        if depthPfname!='':
            outfname=depthPfname
        elif outdir=='':
            outfname='./'+dirPFX+self.name+dirSFX+'/'+self.depthPfname
        elif dirPFX==None or dirSFX==None:
            outfname=outdir+'/'+self.depthPfname
        else:
            outfname=outdir+'/'+dirPFX+self.name+dirSFX+'/'+self.depthPfname
            
        if os.path.isfile(outfname):
            shutil.copyfile(outfname, outfname+'_backup');
        np.savetxt(outfname, self.depthP, fmt='%g');
        return;
    
    def VProfileExtend(self, outdir, ak135Arr, depthInter, maxdepth):
        depth=self.depthP[:,0];
        Vs=self.depthP[:,1];
        Vp=self.depthP[:,2];
        Rou=self.depthP[:,3];
        try:
            Q=self.depthP[:,4];
            Qflag=True;
        except:
            Qflag=False;
        # Interpolate to depthInter array
        VpInter=np.interp(depthInter, depth, Vp);
        VsInter=np.interp(depthInter, depth, Vs);
        RouInter=np.interp(depthInter, depth, Rou);
        if Qflag==True:
            QInter=np.interp(depthInter, depth, Q);
        # Extend to ak135 410km with linear trend
        depthak135=ak135Arr[:,0];
        RouAk135=ak135Arr[:,1];
        VpAk135=ak135Arr[:,2];
        VsAk135=ak135Arr[:,3];
        QAk135=ak135Arr[:,5];
        if Qflag==False:
            QInter=np.interp(depthInter, depthak135, QAk135);
        Vsak135_410=4.8702;
        Vpak135_410=9.0302;
        Rouak135_410=3.5068;
        Qak135_410=146.57;
        depInter2=210.+np.arange((410.-210.)/10.+1)*10.;
        
        Vsak135_2=np.append(Vs[-1], Vsak135_410);
        Vpak135_2=np.append(Vp[-1], Vpak135_410);
        Rouak135_2=np.append(Rou[-1], Rouak135_410);
        if Qflag==True:
            Qak135_2=np.append(Q[-1], Qak135_410);
        else:
            Qak135_2=np.append(QInter[-1], Qak135_410);
        depthak135_2=np.append(depth[-1], 410.)
        
        VsInter2=np.interp(depInter2, depthak135_2, Vsak135_2);
        VpInter2=np.interp(depInter2, depthak135_2, Vpak135_2);
        RouInter2=np.interp(depInter2, depthak135_2, Rouak135_2);
        QInter2=np.interp(depInter2, depthak135_2, Qak135_2);
        
        depthInter=np.append(depthInter, depInter2);
        VsInter=np.append(VsInter,VsInter2);
        VpInter=np.append(VpInter,VpInter2);
        RouInter=np.append(RouInter,RouInter2);
        QInter=np.append(QInter,QInter2);
        # Interpolate with 
        Vsak135_3=VsAk135[(depthak135>410.)];
        Vpak135_3=VpAk135[(depthak135>410.)];
        Rouak135_3=RouAk135[(depthak135>410.)];
        Qak135_3=QAk135[(depthak135>410.)];
        depthak135_3=depthak135[(depthak135>410.)];
        depInter3=410.+np.arange((maxdepth-410.)/50.)*50.+50.;
        
        VsInter3=np.interp(depInter3, depthak135_3, Vsak135_3);
        VpInter3=np.interp(depInter3, depthak135_3, Vpak135_3);
        RouInter3=np.interp(depInter3, depthak135_3, Rouak135_3);
        QInter3=np.interp(depInter3, depthak135_3, Qak135_3);
        
        depthInter=np.append(depthInter, depInter3);
        VsInter=np.append(VsInter,VsInter3);
        VpInter=np.append(VpInter,VpInter3);
        RouInter=np.append(RouInter,RouInter3);
        QInter=np.append(QInter,QInter3);
        
        outfname=outdir+'/'+self.name+'_mod';
        L=depthInter.size;
        # print depthInter.size, VsInter.size, VpInter.size, RouInter.size, QInter.size
        outArr=np.append(depthInter, VsInter);
        outArr=np.append(outArr, VpInter);
        outArr=np.append(outArr, RouInter);
        outArr=np.append(outArr, QInter);
        outArr=outArr.reshape((5, L));
        outArr=outArr.T;
        np.savetxt(outfname, outArr, fmt='%g');
        self.depthPExt=outArr;
        return;
         
    
    def LoadGrDisp(self, datadir='', dirPFX='', dirSFX='', GrDispfname=''):
        if GrDispfname!='':
            infname=GrDispfname
        elif datadir=='':
            infname='./'+dirPFX+self.name+dirSFX+'/'+self.GrDispfname
        else:
            infname=datadir+'/'+dirPFX+self.name+dirSFX+'/'+self.GrDispfname
        if os.path.isfile(infname):
            self.DispGr=np.loadtxt(infname)
        else:
            print 'Warning: No Group Vel Dispersion File for:'+str(self.lon)+' '+str(self.lat)
        return;
    
    def SaveGrDisp(self, outdir='', dirPFX='', dirSFX='', GrDispfname=''):
        if GrDispfname!='':
            outfname=GrDispfname
        elif outdir=='':
            outfname='./'+dirPFX+self.name+dirSFX+'/'+self.GrDispfname
            outdir='./'+dirPFX+self.name+dirSFX;
        else:
            outfname=outdir+'/'+dirPFX+self.name+dirSFX+'/'+self.GrDispfname
            outdir=outdir+'/'+dirPFX+self.name+dirSFX
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        np.savetxt(outfname, self.DispGr, fmt='%g')
        return;
    
    def LoadPhDisp(self, datadir='', dirPFX='', dirSFX='', PhDispfname=''):
        if PhDispfname!='':
            infname=PhDispfname
        elif datadir=='':
            infname='./'+dirPFX+self.name+dirSFX+'/'+self.PhDispfname
        else:
            infname=datadir+'/'+dirPFX+self.name+dirSFX+'/'+self.PhDispfname
        if os.path.isfile(infname):
            self.DispPh=np.loadtxt(infname)
        else:
            print 'Warning: No Phase Vel Dispersion File for:'+str(self.lon)+' '+str(self.lat)
        return

    def SavePhDisp(self, outdir='', dirPFX='', dirSFX='', PhDispfname=''):
        if PhDispfname!='':
            outfname=PhDispfname
        elif outdir=='':
            outfname='./'+dirPFX+self.name+dirSFX+'/'+self.PhDispfname;
            outdir='./'+dirPFX+self.name+dirSFX;
        else:
            outfname=outdir+'/'+dirPFX+self.name+dirSFX+'/'+self.PhDispfname;
            outdir=outdir+'/'+dirPFX+self.name+dirSFX;
        if not os.path.isdir(outdir):
            os.makedirs(outdir);

        np.savetxt(outfname, self.DispPh, fmt='%g')
        return;
    
    def PlotDisp(self, xcl=0, xlabel='Period(s)', ycl={int(1):None, int(2):int(3)}, ylabel='Velocity(km/s)', title='', datatype='PhV', ax=None):
        if ax==None:
            ax=plt.subplot()
        if datatype=='PhV':
            Inarray=self.DispPh
            if title=='':
                title='Phase Velocity Dispersion Curve'
        elif datatype=='GrV':
            if title=='':
                title='Group Velocity Dispersion Curve'
            Inarray=self.DispGr
        else:
            print 'Error: Unknow Data Type!'
            return
        if Inarray.size==0:
            print 'Warning: No Dispersion Data for:'+str(self.lon)+' '+str(self.lat)
            return
        X=Inarray[:,xcl]
        for yC in ycl.keys():
            Y=Inarray[:,yC]
            if ycl[yC]==None:
                line1=ax.plot(X, Y, '-',lw=3) #
            else:
                errC=ycl[yC]
                Yerr=Inarray[:,errC]
                plt.errorbar(X, Y, fmt='.', lw=2, yerr=Yerr)       
        ###
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)              
        return
    
    def PlotDispBoth(self, xcl=0, xlabel='Period(s)', ycl={int(1):None, int(2):int(3)}, ylabel='Velocity(km/s)', title='Dispersion Curve', ax=None):
        if ax==None:
            ax=plt.subplot()
        Inarray=self.DispPh
        X=Inarray[:,xcl]
        for yC in ycl.keys():
            Y=Inarray[:,yC]
            if ycl[yC]==None:
                line1, =ax.plot(X, Y, '-b',lw=3) #
            else:
                errC=ycl[yC]
                Yerr=Inarray[:,errC]
                plt.errorbar(X, Y, fmt='.g', lw=2, yerr=Yerr)
        Inarray=self.DispGr
        X=Inarray[:,xcl]
        for yC in ycl.keys():
            Y=Inarray[:,yC]
            if ycl[yC]==None:
                line2, =ax.plot(X, Y, '-k',lw=3) #
            else:
                errC=ycl[yC]
                Yerr=Inarray[:,errC]
                plt.errorbar(X, Y, fmt='.g', lw=2, yerr=Yerr)
        ax.legend([line1, line2], ['Phase V', 'Group V'], loc=0)
        [xmin, xmax, ymin, ymax]=plt.axis()
        plt.axis([xmin-1, xmax+0.5, ymin, ymax])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)              
        return
    
    def PlotVProfile(self, xcl={int(1):None, int(2):None}, xlabel='Velocity(km/s)', ycl=0, ylabel='Depth(km)', title='Depth Profile', depLimit=None, ax=None):
        if ax==None:
            ax=plt.subplot()
        Inarray=self.depthP
        if Inarray.size==0:
            print 'Warning: No Dispersion Data for:'+str(self.lon)+' '+str(self.lat)
            return
        Y=Inarray[:,ycl]
        if depLimit!=None:
            yindex=Y<depLimit
            Y=Y[yindex]
            Inarray=Inarray[:Y.size, :]
        for xC in xcl.keys():
            X=Inarray[:,xC]
            if xcl[xC]==None:
                ax.plot(X, Y, lw=3) #
            else:
                errC=xcl[xC]
                Yerr=Inarray[:,errC]
                plt.errorbar(X, Y, fmt='.g', lw=2, yerr=Yerr)
        plt.xlabel(xlabel)  
        plt.ylabel(ylabel)
        plt.title(title)
        plt.gca().invert_yaxis()
        return
    
    def IsInRegion(self, maxlon=360, minlon=0, maxlat=90, minlat=-90 ):
        if self.lon < maxlon and self.lon > minlon and self.lat < maxlat and self.lat > minlat:
            return True;
        else:
            return False;
    
    def HorizontalExtension(self, avgArr, ParentsLst, ChildLst, AncestorLst, ChildArrLst,\
            dlat, dlon, maxlat, minlat, maxlon, minlon, Dref=500.):
        radius = 6371.1391285;
        lon=self.lon
        lat=self.lat
        PI=math.pi
        lat_temp=math.atan(0.993277 * math.tan(lat/180.*PI))*180./PI;
        dx_km=radius*math.sin( (90.-lat_temp)/180.*PI )*dlon/180.*PI;
        dy_km = radius*dlat/180.*PI;
        newChLst=np.array([]);
        newAnLst=np.array([]);
        dist=dy_km;
        latN=lat+dlat;
        lonN=lon;
        if not (latN<minlat or latN>maxlat or lonN<minlon or lonN > maxlon):
            nameN='%g'%(lonN) + '_%g' %(latN);
            if ParentsLst[ParentsLst==nameN].size==0:
                if ChildLst.size==0:
                    newArr=((Dref-dist)*self.depthP+dist*avgArr)/Dref;
                    newChLst=np.append(newChLst, nameN);
                    nameAn='%g'%(self.user0) + '_%g' %(self.user1);
                    newAnLst=np.append(newAnLst, nameAn);
                    ChildArrLst.append(newArr);
                elif ChildLst[ChildLst==nameN].size==0:
                    newArr=((Dref-dist)*self.depthP+dist*avgArr)/Dref;
                    newChLst=np.append(newChLst, nameN);
                    nameAn='%g'%(self.user0) + '_%g' %(self.user1);
                    newAnLst=np.append(newAnLst, nameAn);
                    ChildArrLst.append(newArr);
                else:
                    newArr=ChildArrLst[np.where(ChildLst==nameN)[0][0]];
                    ChildArrLst[np.where(ChildLst==nameN)[0][0]]=((Dref-dist)*self.depthP+dist*newArr)/Dref;
        dist=dy_km;
        latN=lat-dlat;
        lonN=lon;
        if not (latN<minlat or latN>maxlat or lonN<minlon or lonN > maxlon):
            nameN='%g'%(lonN) + '_%g' %(latN);
            if ParentsLst[ParentsLst==nameN].size==0:
                if ChildLst.size==0:
                    newArr=((Dref-dist)*self.depthP+dist*avgArr)/Dref;
                    newChLst=np.append(newChLst, nameN);
                    nameAn='%g'%(self.user0) + '_%g' %(self.user1);
                    newAnLst=np.append(newAnLst, nameAn);
                    ChildArrLst.append(newArr);
                elif ChildLst[ChildLst==nameN].size==0:
                    newArr=((Dref-dist)*self.depthP+dist*avgArr)/Dref;
                    newChLst=np.append(newChLst, nameN);
                    nameAn='%g'%(self.user0) + '_%g' %(self.user1);
                    newAnLst=np.append(newAnLst, nameAn);
                    ChildArrLst.append(newArr);
                else:
                    newArr=ChildArrLst[np.where(ChildLst==nameN)[0][0]];
                    ChildArrLst[np.where(ChildLst==nameN)[0][0]]=((Dref-dist)*self.depthP+dist*newArr)/Dref;
        dist=dx_km;
        latN=lat;
        lonN=lon-dlon;
        if not (latN<minlat or latN>maxlat or lonN<minlon or lonN > maxlon):
            nameN='%g'%(lonN) + '_%g' %(latN);
            if ParentsLst[ParentsLst==nameN].size==0:
                if ChildLst.size==0:
                    newArr=((Dref-dist)*self.depthP+dist*avgArr)/Dref;
                    newChLst=np.append(newChLst, nameN);
                    nameAn='%g'%(self.user0) + '_%g' %(self.user1);
                    newAnLst=np.append(newAnLst, nameAn);
                    ChildArrLst.append(newArr);
                elif ChildLst[ChildLst==nameN].size==0:
                    newArr=((Dref-dist)*self.depthP+dist*avgArr)/Dref;
                    newChLst=np.append(newChLst, nameN);
                    nameAn='%g'%(self.user0) + '_%g' %(self.user1);
                    newAnLst=np.append(newAnLst, nameAn);
                    ChildArrLst.append(newArr);
                else:
                    newArr=ChildArrLst[np.where(ChildLst==nameN)[0][0]];
                    ChildArrLst[np.where(ChildLst==nameN)[0][0]]=((Dref-dist)*self.depthP+dist*newArr)/Dref;
        dist=dx_km;
        latN=lat;
        lonN=lon+dlon;
        if not (latN<minlat or latN>maxlat or lonN<minlon or lonN > maxlon):
            nameN='%g'%(lonN) + '_%g' %(latN);
            if ParentsLst[ParentsLst==nameN].size==0:
                if ChildLst.size==0:
                    newArr=((Dref-dist)*self.depthP+dist*avgArr)/Dref;
                    newChLst=np.append(newChLst, nameN);
                    nameAn='%g'%(self.user0) + '_%g' %(self.user1);
                    newAnLst=np.append(newAnLst, nameAn);
                    ChildArrLst.append(newArr);
                elif ChildLst[ChildLst==nameN].size==0:
                    newArr=((Dref-dist)*self.depthP+dist*avgArr)/Dref;
                    newChLst=np.append(newChLst, nameN);
                    nameAn='%g'%(self.user0) + '_%g' %(self.user1);
                    newAnLst=np.append(newAnLst, nameAn);
                    ChildArrLst.append(newArr);
                else:
                    newArr=ChildArrLst[np.where(ChildLst==nameN)[0][0]];
                    ChildArrLst[np.where(ChildLst==nameN)[0][0]]=((Dref-dist)*self.depthP+dist*newArr)/Dref;
        return newChLst, newAnLst;
        
    def HorizontalExtensionSearch(self, avgArr, ModelLst, ModelArrLst, dlat, dlon, maxlatM, minlatM, maxlonM, minlonM, Dref=500.):
        
        radius = 6371.1391285;
        lon=self.lon
        lat=self.lat
        PI=math.pi
        lat_temp=math.atan(0.993277 * math.tan(lat/180.*PI))*180./PI;
        dx_km=radius*math.sin( (90.-lat_temp)/180.*PI )*dlon/180.*PI;
        dy_km = radius*dlat/180.*PI;
        # Dref=1000.
        if self.lon >= maxlonM:
            clon=maxlonM;
            searchlon=1;
        elif self.lon <= minlonM:
            clon=minlonM;
            searchlon=2;
        else:
            clon=self.lon;
            searchlon=3;
            
        if self.lat >= maxlatM:
            clat=maxlatM;
            searchlat=1;
        elif self.lat <= minlatM:
            clat=minlatM;
            searchlat=2;
        else:
            clat=self.lat;
            searchlat=3;
        
        slat=clat;
        slon=clon;
        nameS='%g'%(slon) + '_%g' %(slat)
        if searchlat==1:
            while ( ModelLst[ModelLst==nameS].size==0):
                slat=slat-dlat;
                nameS='%g'%(slon) + '_%g' %(slat);
        if searchlat==2:
            while ( ModelLst[ModelLst==nameS].size==0):
                slat=slat+dlat;
                nameS='%g'%(slon) + '_%g' %(slat);
        if searchlon==3 and searchlat==3:
            nameS1=nameS;
            nameS2=nameS;
            slat1=slat;
            slat2=slat;
            while ( ModelLst[ModelLst==nameS1].size==0 and ModelLst[ModelLst==nameS2].size==0):
                slat1=slat1-dlat;
                nameS1='%g'%(slon) + '_%g' %(slat1);
                slat2=slat2+dlat;
                nameS2='%g'%(slon) + '_%g' %(slat2);
            if  ModelLst[ModelLst==nameS2].size==0:
                nameS=nameS1;
                slat=slat1;
            elif ModelLst[ModelLst==nameS1].size==0:
                nameS=nameS2;
                slat=slat2;
            else:
                Dlat=slat2-slat;
                searchlat=4;
        
        if slat>clat:
            Ulat=slat;
            Llat=clat;
        else:
            Ulat=clat;
            Llat=slat;
        
        slat=clat;
        slon=clon;
        nameS='%g'%(slon) + '_%g' %(slat)
        if searchlon==1:
            while ( ModelLst[ModelLst==nameS].size==0):
                slon=slon-dlon;
                nameS='%g'%(slon) + '_%g' %(slat);
        if searchlon==2:
            while ( ModelLst[ModelLst==nameS].size==0):
                slon=slon+dlon;
                nameS='%g'%(slon) + '_%g' %(slat);
        if searchlon==3 and searchlat==3:
            nameS1=nameS;
            nameS2=nameS;
            slon1=slon;
            slon2=slon;
            while ( ModelLst[ModelLst==nameS1].size==0 and ModelLst[ModelLst==nameS2].size==0):
                slon1=slon1-dlon;
                nameS1='%g'%(slon1) + '_%g' %(slat);
                slon2=slon2+dlon;
                nameS2='%g'%(slon2) + '_%g' %(slat);
            if  ModelLst[ModelLst==nameS1].size==0:
                nameS=nameS2;
                slon=slon2;
            elif ModelLst[ModelLst==nameS2].size==0:
                nameS=nameS1;
                slon=slon1;
            else:
                Dlon=slon2-slon;
                searchlon=4;
            
            
        if slon>clon:
            Ulon=slon;
            Llon=clon;
        else:
            Ulon=clon;
            Llon=slon;

        if searchlat==3 and (searchlon==1 or searchlon==2):
            mlat=lat;
            mlon=slon;
            dist, az, baz=obsGeo.gps2DistAzimuth(mlat, mlon, lat, lon ) # distance is in m
            dist=dist/1000;
            nameM='%g'%(mlon) + '_%g' %(mlat);
            Arr=ModelArrLst[np.where(ModelLst==nameM)[0][0]];
            self.depthP=((Dref-dist)*Arr+dist*avgArr)/Dref;
            sfx='_'+nameM;
            self.SetVProfileFname(prefix='', suffix=sfx);
            return;
        elif searchlon==3 and (searchlat==1 or searchlat==2):
            mlat=slat;
            mlon=lon;
            dist, az, baz=obsGeo.gps2DistAzimuth(mlat, mlon, lat, lon ) # distance is in m
            dist=dist/1000;
            nameM='%g'%(mlon) + '_%g' %(mlat);
            try:
                Arr=ModelArrLst[np.where(ModelLst==nameM)[0][0]];
            except IndexError:
                print nameM, lon, lat;
                raise IndexError('INDEX')
            self.depthP=((Dref-dist)*Arr+dist*avgArr)/Dref;
            sfx='_'+nameM;
            self.SetVProfileFname(prefix='', suffix=sfx);
            return;  
        elif searchlat==4 and searchlon==4:
            if dx_km*Dlon>dy_km*Dlat:
                name1='%g'%(lon) + '_%g' %(clat-Dlat);
                name2='%g'%(lon) + '_%g' %(clat+Dlat);
                Arr1=ModelArrLst[np.where(ModelLst==name1)[0][0]];
                Arr2=ModelArrLst[np.where(ModelLst==name2)[0][0]];
                self.depthP=npr.evaluate('0.5*Arr1+0.5*Arr2');
                sfx='_'+name1+'_'+name2;
                self.SetVProfileFname(prefix='', suffix=sfx);
                return;
            else:
                name1='%g'%(lon-Dlon) + '_%g' %(clat);
                name2='%g'%(lon+Dlon) + '_%g' %(clat);
                Arr1=ModelArrLst[np.where(ModelLst==name1)[0][0]];
                Arr2=ModelArrLst[np.where(ModelLst==name2)[0][0]];
                self.depthP=npr.evaluate('0.5*Arr1+0.5*Arr2');
                sfx='_'+name1+'_'+name2;
                self.SetVProfileFname(prefix='', suffix=sfx);
                return;
        elif searchlat==4 and searchlon==3:
            if dx_km*(abs(slon-lon))>dy_km*Dlat:
                name1='%g'%(lon) + '_%g' %(clat-Dlat);
                name2='%g'%(lon) + '_%g' %(clat+Dlat);
                Arr1=ModelArrLst[np.where(ModelLst==name1)[0][0]];
                Arr2=ModelArrLst[np.where(ModelLst==name2)[0][0]];
                self.depthP=npr.evaluate('0.5*Arr1+0.5*Arr2');
                sfx='_'+name1+'_'+name2;
                self.SetVProfileFname(prefix='', suffix=sfx);
                return;
            else:
                nameM='%g'%(slon) + '_%g' %(lat);
                self.depthP=ModelArrLst[np.where(ModelLst==nameM)[0][0]];
                sfx='_'+nameM;
                self.SetVProfileFname(prefix='', suffix=sfx);
                return;
        elif searchlat==3 and searchlon==4:
            if dy_km*(abs(slat-lat))>dx_km*Dlon:
                name1='%g'%(lon-Dlon) + '_%g' %(lat);
                name2='%g'%(lon+Dlon) + '_%g' %(lat);
                Arr1=ModelArrLst[np.where(ModelLst==name1)[0][0]];
                Arr2=ModelArrLst[np.where(ModelLst==name2)[0][0]];
                self.depthP=npr.evaluate('0.5*Arr1+0.5*Arr2');
                sfx='_'+name1+'_'+name2;
                self.SetVProfileFname(prefix='', suffix=sfx);
                return;
            else:
                nameM='%g'%(lon) + '_%g' %(slat);
                self.depthP=ModelArrLst[np.where(ModelLst==nameM)[0][0]];
                sfx='_'+nameM;
                self.SetVProfileFname(prefix='', suffix=sfx);
                return;
        elif searchlat==3 and searchlon==3:
            if dy_km*(abs(slat-lat))>dx_km*(abs(slon-lon)):
                nameM='%g'%(slon) + '_%g' %(lat);
                self.depthP=ModelArrLst[np.where(ModelLst==nameM)[0][0]];
                sfx='_'+nameM;
                self.SetVProfileFname(prefix='', suffix=sfx);
                return;
            else:
                nameM='%g'%(lon) + '_%g' %(slat);
                self.depthP=ModelArrLst[np.where(ModelLst==nameM)[0][0]];
                sfx='_'+nameM;
                self.SetVProfileFname(prefix='', suffix=sfx);
                return;
        lonarr=np.arange((Ulon-Llon)/dlon)*dlon+Llon;
        latarr=np.arange((Ulat-Llat)/dlat)*dlat+Llat;
        distM=9999;
        Mlat=0;
        Mlon=0;
        for mlon in lonarr:
            for mlat in latarr:
                if searchlon==1 and searchlat==1:
                    slope=(Llat-Ulat)/(Ulon-Llon);
                    klat=(mlon-Llon)*slope+Ulat;
                    if mlat<klat:
                        continue;
                    name='%g'%(mlon) + '_%g' %(mlat);
                    if ModelLst[ModelLst==name].size==0:
                        continue;
                    dist, az, baz=obsGeo.gps2DistAzimuth(mlat, mlon, lat, lon ) # distance is in m
                    dist=dist/1000;
                    if distM>dist:
                        Mlat=mlat;
                        Mlon=mlon;
                if searchlon==1 and searchlat==2:
                    slope=(Ulat-Llat)/(Ulon-Llon);
                    klat=(mlon-Llon)*slope+Llat;
                    if mlat>klat:
                        continue;
                    name='%g'%(mlon) + '_%g' %(mlat);
                    if ModelLst[ModelLst==name].size==0:
                        continue;
                    dist, az, baz=obsGeo.gps2DistAzimuth(mlat, mlon, lat, lon ) # distance is in m
                    dist=dist/1000;
                    if distM>dist:
                        Mlat=mlat;
                        Mlon=mlon;
                if searchlon==2 and searchlat==1:
                    slope=(Ulat-Llat)/(Ulon-Llon);
                    klat=(mlon-Llon)*slope+Llat;
                    if mlat<klat:
                        continue;
                    name='%g'%(mlon) + '_%g' %(mlat);
                    if ModelLst[ModelLst==name].size==0:
                        continue;
                    dist, az, baz=obsGeo.gps2DistAzimuth(mlat, mlon, lat, lon ) # distance is in m
                    dist=dist/1000;
                    if distM>dist:
                        Mlat=mlat;
                        Mlon=mlon;
                if searchlon==2 and searchlat==2:
                    slope=(Llat-Ulat)/(Ulon-Llon);
                    klat=(mlon-Llon)*slope+Ulat;
                    if mlat>klat:
                        continue;
                    name='%g'%(mlon) + '_%g' %(mlat);
                    if ModelLst[ModelLst==name].size==0:
                        continue;
                    dist, az, baz=obsGeo.gps2DistAzimuth(mlat, mlon, lat, lon ) # distance is in m
                    dist=dist/1000;
                    if distM>dist:
                        Mlat=mlat;
                        Mlon=mlon;
        nameM='%g'%(Mlon) + '_%g' %(Mlat);
        Arr=ModelArrLst[np.where(ModelLst==nameM)[0][0]];
        self.depthP=((Dref-dist)*Arr+dist*avgArr)/Dref;
        sfx='_'+nameM;
        self.SetVProfileFname(prefix='', suffix=sfx);
        return;
     
    def Vs2VpRho(self, flag=0, Vsmin=999 ):
        depth=self.depthP[:,0];
        Vs=self.depthP[:,1];
        
        if Vsmin!=999:
            LArr=Vs<Vsmin;
            if LArr.size!=0:
                UArr=Vs>=Vsmin;
                Vs1=Vsmin*LArr;
                Vs2=Vs*UArr;
                Vs=npr.evaluate('Vs1+Vs2');
        
        if flag==0:
            eqIndex=np.ediff1d(depth);
            mohoDepth=(depth[eqIndex==0])[-1];
            Vs_crust=Vs[depth<=mohoDepth][:-1];
            Vs_mantle=Vs[depth>=mohoDepth][1:];
            
            Vp_crust, Rho_crust=BrocherCrustVs2VpRho(Vs_crust);
            Vp_mantle, Rho_mantle=MantleVs2VpRho(Vs_mantle);
            Vp=np.append(Vp_crust, Vp_mantle);
            Rho=np.append(Rho_crust, Rho_mantle);
        elif flag==1:
            Vp, Rho=MantleVs2VpRho(Vs);
            
        outArr=np.append(depth, Vs);
        outArr=np.append(outArr, Vp);
        outArr=np.append(outArr, Rho);
        outArr=outArr.reshape((4, Vs.size))
        outArr=outArr.T;
        self.depthP=outArr;
        return;
    
    
class GeoMap(object):
    """
    A object contains a list of GeoPoint
    """
    def __init__(self,geopoints=None):
        self.geopoints=[]
        if isinstance(geopoints, GeoPoint):
            geopoints = [geopoints]
        if geopoints:
            self.geopoints.extend(geopoints)

    def __add__(self, other):
        """
        Add two GeoMaps with self += other.
        """
        if isinstance(other, GeoPoint):
            other = GeoMap([other])
        if not isinstance(other, GeoMap):
            raise TypeError
        geopoints = self.geopoints + other.geopoints
        return self.__class__(geopoints=geopoints)

    def __len__(self):
        """
        Return the number of GeoPoints in the GeoMap object.
        """
        return len(self.geopoints)

    def __getitem__(self, index):
        """
        __getitem__ method of GeoMap objects.
        :return: GeoPoint objects
        """
        if isinstance(index, slice):
            return self.__class__(geopoints=self.geopoints.__getitem__(index))
        else:
            return self.geopoints.__getitem__(index)

    def append(self, geopoint):
        """
        Append a single GeoPoint object to the current GeoMap object.
        """
        if isinstance(geopoint, GeoPoint):
            self.geopoints.append(geopoint)
        else:
            msg = 'Append only supports a single GeoPoint object as an argument.'
            raise TypeError(msg)
        return self
    
    def ReadGeoMapLst(self, mapfile ):
        """
        Read GeoPoint List from a txt file
        name longitude latitude
        """
        f = open(mapfile, 'r')
        Name=[]
        for lines in f.readlines():
            lines=lines.split()
            name=lines[0]
            lon=float(lines[1])
            lat=float(lines[2])
            if Name.__contains__(name):
                index=Name.index(name)
                if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                    raise ValueError('Incompatible GeoPoint Location:' + netsta+' in GeoPoint List!')
                else:
                    print 'Warning: Repeated GeoPoint:' +name+' in GeoPoint List!'
                    continue
            Name.append(name)
            self.append(GeoPoint (name=name, lon=lon, lat=lat))
            f.close()
        return
    
    
    def Trim(self, maxlon=360, minlon=0, maxlat=90, minlat=-90):
        TrimedGeoMap=GeoMap()
        for geopoint in self.geopoints:
            if geopoint.IsInRegion(maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat):
                TrimedGeoMap.append(geopoint)
        return TrimedGeoMap
    
    def GetINTGeoMap(self):
        NewGeoMap=GeoMap();
        for geopoint in self.geopoints:
            if abs(geopoint.lon-int(geopoint.lon)) > 0.1 or abs(geopoint.lat-int(geopoint.lat))>0.1:
                continue;
            geopoint.SetName()
            NewGeoMap.append(geopoint);
        # print 'End of Converting SES3D seismograms to SAC files !'
        return NewGeoMap;
    
    
    def SetAllfname(self):
        for geopoint in self.geopoints:
            geopoint.SetAllfname()
        return
    
    def SetPhDispfname(self, prefix='MC.', suffix='.acc.average.p.disp'):
        for geopoint in self.geopoints:
            geopoint.SetPhDispfname(prefix=prefix, suffix=suffix)
        return
    
    def SetGrDispfname(self, prefix='MC.', suffix='.acc.average.p.disp'):
        for geopoint in self.geopoints:
            geopoint.SetGrDispfname(prefix=prefix, suffix=suffix)
        return
    
    def SetVProfileFname(self, prefix='MC.', suffix='.acc.average.mod.q'):
        for geopoint in self.geopoints:
            geopoint.SetVProfileFname(prefix=prefix, suffix=suffix)
        return
    
    def LoadVProfile(self, datadir='', dirPFX='', dirSFX='', depthPfname=''):
        
        for geopoint in self.geopoints:
            geopoint.LoadVProfile(datadir=datadir, dirPFX=dirPFX, dirSFX=dirSFX, depthPfname=depthPfname)
        return
    
    def SaveVProfile(self, outdir='', dirPFX='', dirSFX='', depthPfname=''):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        for geopoint in self.geopoints:
            geopoint.SaveVProfile(outdir=outdir, dirPFX=dirPFX, dirSFX=dirSFX, depthPfname=depthPfname)
        return
    
    def ReplaceVProfile(self, outdir='', dirPFX='', dirSFX='', depthPfname=''):
        for geopoint in self.geopoints:
            geopoint.ReplaceVProfile(outdir=outdir, dirPFX=dirPFX, dirSFX=dirSFX, depthPfname=depthPfname)
        return
    
    def LoadGrDisp(self, datadir='', dirPFX='', dirSFX='', GrDispfname=''):
        for geopoint in self.geopoints:
            geopoint.LoadGrDisp(datadir=datadir, dirPFX=dirPFX, GrDispfname=GrDispfname)
        return
    
    def LoadPhDisp(self, datadir='', dirPFX='', dirSFX='', PhDispfname=''):
        for geopoint in self.geopoints:
            geopoint.LoadPhDisp(datadir=datadir, dirPFX=dirPFX, PhDispfname=PhDispfname)
        return
    
    def GetGeoMapfromDir(self, datadir='', dirPFX='', dirSFX=''):
        pattern=datadir+'/*';
        LonLst=np.array([]);
        LatLst=np.array([]);
        for dirname in sorted(glob.glob(pattern)):
            dirname=dirname.split('/');
            dirname=dirname[len(dirname)-1];
            if dirPFX!='':
                dirname=dirname.split(dirPFX)[1];
            if dirSFX!='':
                dirname=dirname.split(dirSFX);
                if len(dirname) > 2:
                    raise ValueError('Directory Suffix Error!');
                dirname=dirname[0];
            lon=dirname.split('_')[0];
            lat=dirname.split('_')[1];
            LonLst=np.append(LonLst, float(lon));
            LatLst=np.append(LatLst, float(lat));
        indLst=np.lexsort((LonLst,LatLst));
        for indS in indLst:
            lon='%g' %(LonLst[indS])
            lat='%g' %(LatLst[indS])
            self.append(GeoPoint (name=lon+'_'+lat, lon=float(lon), lat=float(lat)));
        return
    
    def GetMapLimit(self):
        minlon=360.
        maxlon=0.
        minlat=90.
        maxlat=-90.
        for geopoint in self.geopoints:
            if geopoint.lon > maxlon:
                maxlon=geopoint.lon
            if geopoint.lon < minlon:
                minlon=geopoint.lon
            if geopoint.lat > maxlat:
                maxlat=geopoint.lat
            if geopoint.lat < minlat:
                minlat=geopoint.lat
        self.minlon=minlon
        self.maxlon=maxlon
        self.minlat=minlat
        self.maxlat=maxlat
        return
     
    def BrowseFigures(self, outdir='', dirPFX='', dirSFX='', datatype='All', depLimit=None, \
                      llcrnrlon=None, llcrnrlat=None,urcrnrlon=None,urcrnrlat=None, browseflag=True, saveflag=False):
        
        if outdir!='' and not os.path.isdir(outdir):
            os.makedirs(outdir);
        if llcrnrlon==None or llcrnrlat==None or urcrnrlon==None or urcrnrlat==None:
            llcrnrlon=self.minlon
            llcrnrlat=self.minlat
            urcrnrlon=self.maxlon
            urcrnrlat=self.maxlat
            # print llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat
        for geopoint in self.geopoints:
            print 'Plotting:'+geopoint.name
            if geopoint.depthP.size==0 and geopoint.DispGr.size==0 and geopoint.DispPh.size==0:
                continue
            plt.close('all')
            fig=plb.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
            if datatype=='All':
                fig.add_subplot(3,1,1)
            else:
                fig.add_subplot(2,1,1)
            m = Basemap(llcrnrlon=llcrnrlon-1, llcrnrlat=llcrnrlat-1, urcrnrlon=urcrnrlon+1, urcrnrlat=urcrnrlat+1, \
                rsphere=(6378137.00,6356752.3142), resolution='l', projection='merc')
            lon = geopoint.lon
            lat = geopoint.lat
            x,y = m(lon, lat)
            m.plot(x, y, 'ro', markersize=5)
            m.drawcoastlines()
            m.etopo()
            # m.fillcontinents()
            # draw parallels
            m.drawparallels(np.arange(-90,90,10),labels=[1,1,0,1])
            # draw meridians
            m.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])
            plt.title('Longitude:'+str(geopoint.lon)+' Latitude:'+str(geopoint.lat), fontsize=15)
            if datatype=='All':
                geopoint.PlotDispBoth(ax=plt.subplot(3,1,2))
                # geopoint.PlotDisp(datatype='GrV', ax=plt.subplot(3,1,2))
                geopoint.PlotVProfile(ax=plt.subplot(3,1,3), depLimit=depLimit)
            elif datatype=='DISP':
                # geopoint.PlotDisp(datatype='PhV',ax=plt.subplot(3,1,2))
                # geopoint.PlotDisp(datatype='GrV',ax=plt.subplot(3,1,2))
                geopoint.PlotDispBoth(ax=plt.subplot(3,1,2))
            elif datatype=='VPr':
                geopoint.PlotVProfile(depLimit=depLimit,ax=plt.subplot(2,1,2))
            else:
                raise ValueError('Unknown datatype')
            # fig.suptitle('Longitude:'+str(geopoint.lon)+' Latitude:'+str(geopoint.lat), fontsize=15)
            if browseflag==True:
                plt.draw()
                plt.pause(1) # <-------
                raw_input("<Hit Enter To Close>")
                plt.close('all')
            if saveflag==True and outdir!='':
                if dirPFX==None or dirSFX==None:
                    fig.savefig(outdir+'/'+datatype+'_'+geopoint.name+'.ps', format='ps');
                else:
                    fig.savefig(outdir+'/'+dirPFX+geopoint.name+dirSFX+'/'+datatype+'_'+geopoint.name+'.ps', format='ps')
        return;

    def SavePhDisp(self, outdir='', dirPFX='', dirSFX='', PhDispfname=''):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for geopoint in self.geopoints:
            geopoint.SavePhDisp(outdir=outdir, dirPFX=dirPFX, dirSFX=dirSFX, PhDispfname=PhDispfname);
        print 'End of Saving Phave V Points!';
        return;
    
    def SaveGrDisp(self, outdir='', dirPFX='', dirSFX='', GrDispfname=''):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for geopoint in self.geopoints:
            geopoint.SaveGrDisp(outdir=outdir, dirPFX=dirPFX, dirSFX=dirSFX, GrDispfname=GrDispfname);
        print 'End of Saving Group V Points!';
        return;
    
    def VProfileExtend(self, outdir, ak135fname, depthInter, maxdepth):
        ak135Arr=np.loadtxt(ak135fname);
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        for geopoint in self.geopoints:
            geopoint.VProfileExtend(outdir=outdir, ak135Arr=ak135Arr, depthInter=depthInter, maxdepth=maxdepth);
        print 'End of Extension of Vertical Profile!';
        return;
        
    def VProfileExtendParallel(self, outdir, ak135fname, depthInter, maxdepth):
        ak135Arr=np.loadtxt(ak135fname);
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        VProfExtend=partial(geoPEXTEND, outdir=outdir, ak135Arr=ak135Arr, depthInter=depthInter, maxdepth=maxdepth); 
        pool = mp.Pool()
        pool.map(VProfExtend, self.geopoints) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Extension of Vertical Profile(Parallel)!';
        return;
    
    def GetAvgVProfile(self, outdir, outfname='avg.mod'):
        outArr=self.geopoints[0].depthPExt;
        L=1;
        for geopoint in self.geopoints[1:]:
            print geopoint.name
            outArr=outArr+geopoint.depthPExt;
            L=L+1;
        outArr=outArr/L;
        outfname=outdir+'/'+outfname;
        np.savetxt(outfname, outArr, fmt='%g');
        self.avgArr=outArr;
        return outArr;
        
    def HorizontalExtension(self, avgfname, outdir, dlat, dlon, maxlat, minlat, maxlon, minlon):
        Lext=0;
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        oLst=np.array([]);
        avgArr=np.loadtxt(avgfname);
        for geopoint in self.geopoints:
            oLst=np.append(oLst, geopoint.name);
            geopoint.user0=geopoint.lon;
            geopoint.user1=geopoint.lat;
        
        ChildLst=np.array([]);
        AncestorLst=np.array([]);
        ChildArrLst=[];
        for geopoint in self.geopoints:
            appChLst, appAnLst=geopoint.HorizontalExtension(avgArr, ParentsLst=oLst, ChildLst=ChildLst,\
                    AncestorLst=AncestorLst, ChildArrLst=ChildArrLst, dlat=dlat, dlon=dlon, \
                    maxlat=maxlat, minlat=minlat, maxlon=maxlon, minlon=minlon);
            ChildLst=np.append(ChildLst, appChLst);
            AncestorLst=np.append(AncestorLst, appAnLst);
        newGeomap=GeoMap();
        for i in np.arange(ChildLst.size):
            tempNewGeoP=GeoPoint()
            tempNewGeoP.name=ChildLst[i]
            templon, templat=tempNewGeoP.name.split('_');
            tempNewGeoP.lon=float(templon);
            tempNewGeoP.lat=float(templat);
            tempNewGeoP.depthP=ChildArrLst[i];
            tempAnlon, tempAnlat=AncestorLst[i].split('_');
            tempNewGeoP.user0=float(tempAnlon);
            tempNewGeoP.user1=float(tempAnlat);
            outfname=outdir+'/'+tempNewGeoP.name+'_ext_'+AncestorLst[i]+'_1';
            tempNewGeoP.SaveVProfile(depthPfname=outfname);
            Lext=Lext+1;
            newGeomap.append(tempNewGeoP);
        NumGen=1;
        PLst=oLst;
        while (ChildLst.size!=0 and Lext<13000):
            print 'Number of Generations: ', NumGen;
            PLst=np.append(PLst, ChildLst);
            # print PLst[PLst=='133.5_30.5'];
            ChildLst=np.array([]);
            AncestorLst=np.array([]);
            ChildArrLst=[];
            # for geopoint in newGeomap.geopoints:
            #     PLst=np.append(PLst, geopoint.name);
            for geopoint in newGeomap.geopoints:
                appChLst, appAnLst=geopoint.HorizontalExtension(avgArr, ParentsLst=PLst, ChildLst=ChildLst,\
                    AncestorLst=AncestorLst, ChildArrLst=ChildArrLst, dlat=dlat, dlon=dlon, maxlat=maxlat,\
                    minlat=minlat, maxlon=maxlon, minlon=minlon);
                ChildLst=np.append(ChildLst, appChLst);
                AncestorLst=np.append(AncestorLst, appAnLst);
            newGeomap=GeoMap();
            NumGen=NumGen+1
            for i in np.arange(ChildLst.size):
                tempNewGeoP=GeoPoint()
                tempNewGeoP.name=ChildLst[i]
                templon, templat=tempNewGeoP.name.split('_');
                tempNewGeoP.lon=float(templon);
                tempNewGeoP.lat=float(templat);
                tempNewGeoP.depthP=ChildArrLst[i];
                tempAnlon, tempAnlat=AncestorLst[i].split('_');
                tempNewGeoP.user0=float(tempAnlon);
                tempNewGeoP.user1=float(tempAnlat);
                outfname=outdir+'/'+tempNewGeoP.name+'_ext_'+AncestorLst[i]+'_'+str(NumGen);
                tempNewGeoP.SaveVProfile(depthPfname=outfname);
                Lext=Lext+1;
                newGeomap.append(tempNewGeoP);
        print 'End of Extension of Vertical Profile!';
        return;
    
    def HorizontalExtensionS(self, avgfname, outdir, dlat, dlon, maxlat, minlat, maxlon, minlon, maxlatM, minlatM, maxlonM, minlonM):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        oLst=np.array([]);
        avgArr=np.loadtxt(avgfname);
        ModelLst=np.array([]);
        ModelArrLst=[];
        for geopoint in self.geopoints:
            ModelLst=np.append(ModelLst, geopoint.name);
            ModelArrLst.append(geopoint.depthP);
        lonarr=np.arange((maxlon-minlon)/dlon)*dlon+minlon;
        latarr=np.arange((maxlat-minlat)/dlat)*dlat+minlat;
        for Nlon in lonarr:
            for Nlat in latarr:
                nameN='%g'%(Nlon) + '_%g' %(Nlat);
                if ModelLst[ModelLst==nameN].size!=0:
                    continue;
                print 'Extending to ', nameN
                newGeoP=GeoPoint();
                newGeoP.lon=Nlon;
                newGeoP.lat=Nlat;
                newGeoP.HorizontalExtensionSearch(avgArr, ModelLst, ModelArrLst, dlat, dlon, maxlatM, minlatM, maxlonM, minlonM)
                newGeoP.SaveVProfile(outdir=outdir, dirSFX=None);
        return;
    
    def HExtrapolationNearest(self, avgfname, outdir, dlat, dlon, maxlat, minlat, maxlon, minlon, Dref=500.):
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        avgArr=np.loadtxt(avgfname);
        ModelLst=np.array([]);
        for geopoint in self.geopoints:
            ModelLst=np.append(ModelLst, geopoint.name);
        lonarr=np.arange((maxlon-minlon)/dlon+1)*dlon+minlon;
        latarr=np.arange((maxlat-minlat)/dlat+1)*dlat+minlat;
        for Nlon in lonarr:
            for Nlat in latarr:
                mindist=99999;
                mlon=0;
                mlat=0;
                nameN='%g'%(Nlon) + '_%g' %(Nlat);
                if ModelLst[ModelLst==nameN].size!=0:
                    continue;
                print 'Extending to ', nameN
                for geopoint in self.geopoints:
                    dist=great_circle((geopoint.lat, geopoint.lon),(Nlat, Nlon)).km
                    # dist, az, baz=obsGeo.gps2DistAzimuth(geopoint.lat, geopoint.lon, Nlat, Nlon ) # distance is in m
                    # dist=dist/1000;
                    if dist<mindist:
                        mindist=dist;
                        mlon=geopoint.lon;
                        mlat=geopoint.lat;
                        CArr=geopoint.depthP
                newGeoP=GeoPoint();
                newGeoP.lon=Nlon;
                newGeoP.lat=Nlat;
                if mindist < Dref:
                    newGeoP.depthP=((Dref-mindist)*CArr+mindist*avgArr)/Dref;
                else:
                    newGeoP.depthP=avgArr;
                nameM='%g'%(mlon) + '_%g' %(mlat);
                # sfx='_'+nameM;
                sfx='_mod';
                newGeoP.SetVProfileFname(prefix='', suffix=sfx);
                newGeoP.SaveVProfile(outdir=outdir, dirSFX=None);
        return 
    
    def HExtrapolationNearestParallel(self, avgfname, outdir, dlat, dlon, maxlat, minlat, maxlon, minlon, Dref=500.):
        # Dref=1000.;
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        avgArr=np.loadtxt(avgfname);
        ModelLst=np.array([]);
        for geopoint in self.geopoints:
            ModelLst=np.append(ModelLst, geopoint.name);
        lonarr=np.arange((maxlon-minlon)/dlon+1)*dlon+minlon;
        latarr=np.arange((maxlat-minlat)/dlat+1)*dlat+minlat;
        positionArr=[]
        for Nlon in lonarr:
            for Nlat in latarr:
                positionArr.append(np.array([Nlon, Nlat]));
        HEXTRAPOLATE = partial(PositionHExtrapolate, avgArr=avgArr, ModelLst=ModelLst, outdir=outdir, geomap=self, Dref=Dref);
        pool =mp.Pool()
        pool.map(HEXTRAPOLATE, positionArr) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        print 'End of Adding Horizontal Slowness  ( Parallel ) !'
        return;
    
    def FindNearestPoint(self, lon, lat):
        mindist=99999;
        mlon=0;
        mlat=0;
        for geopoint in self.geopoints:
            dist, az, baz=obsGeo.gps2DistAzimuth(geopoint.lat, geopoint.lon, lat, lon ) # distance is in m
            dist=dist/1000;
            if dist<mindist:
                mindist=dist;
                mlon=geopoint.lon;
                mlat=geopoint.lat;
        return mlon, mlat
    
    def GetMinVs(self):
        Vsmin=10;
        lonmin=0;
        latmin=0
        for geopoint in self.geopoints:
            Vs=geopoint.depthP[:,1];
            cVsmin=Vs.min();
            if cVsmin<Vsmin:
                Vsmin=cVsmin;
                lonmin=geopoint.lon;
                latmin=geopoint.lat;
        print 'Minumum Shear Wave Velocity: ', Vsmin,' km/sec at longitude:', lonmin, ' latitude: ', latmin;
        return Vsmin, lonmin, latmin;
    
    def StationDistribution(self, evlo, evla, maxlat, minlat, maxlon, minlon, mapflag='regional_ortho', mapfactor=4, res='i'):
        """
        Read Sation List from a txt file
        stacode longitude latidute network
        """
        Sta=[]
        LonLst=np.array([])
        LatLst=np.array([])
        for geopoint in self.geopoints:
            LonLst=np.append(LonLst, geopoint.lon);
            LatLst=np.append(LatLst, geopoint.lat);
        lon_min=minlon-1
        lat_min=minlat
        lon_max=maxlon
        lat_max=maxlat
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
        # ax=plt.subplot(111)
        # m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, \
        #     rsphere=(6378137.00,6356752.3142), resolution='c', projection='merc')
        if mapflag=='global':
            m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif mapflag=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
            m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/mapfactor)
            # labels = [left,right,top,bottom]
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,0])
            m.drawmeridians(np.arange(-170.0,170.0,10.0))	
        elif mapflag=='regional_merc':
            m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
            m.drawparallels(np.arange(np.round(lat_min),np.round(lat_max),d_lat),labels=[1,0,0,1])
            m.drawmeridians(np.arange(np.round(lon_min),np.round(lon_max),d_lon),labels=[1,0,0,1])
        lon = LonLst
        lat = LatLst
        # x,y = m(lon, lat)
        # m.plot(x, y, 'r^', markersize=2)
        m.drawcoastlines()
        evx, evy=m(evlo, evla)
        m.plot(evx, evy, 'r*', markersize=25)
        # InArr=np.loadtxt('/projects/life9360/station_map/ETOPO1_Ice_g_int.xyz')
        # etopo=InArr[:,2];
        # lons=InArr[:,0];
        # lats =InArr[:,1];
        # lats=lats[lats>0];
        # lons=lons[lats>0];
        # etopo=etopo[lats>0]
        # print 'End of Reading Data'
        # ddir='/projects/life9360/software/basemap-1.0.7/examples';
        # etopo = np.loadtxt(ddir+'/etopo20data.gz')
        # lons  = np.loadtxt(ddir+'/etopo20lons.gz')
        # lats  = np.loadtxt(ddir+'/etopo20lats.gz')
        # url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc'
        blon=np.arange(100)*(maxlon-minlon)/100.+minlon;
        blat=np.arange(100)*(maxlat-minlat)/100.+minlat;
        Blon=blon;
        Blat=np.ones(Blon.size)*minlat;
        x,y = m(Blon, Blat)
        m.plot(x, y, 'r-', lw=3)
        
        Blon=blon;
        Blat=np.ones(Blon.size)*maxlat;
        x,y = m(Blon, Blat)
        m.plot(x, y, 'r-', lw=3)
        
        Blon=np.ones(Blon.size)*minlon;
        Blat=blat;
        x,y = m(Blon, Blat)
        m.plot(x, y, 'r-', lw=3)
        
        Blon=np.ones(Blon.size)*maxlon;
        Blat=blat;
        x,y = m(Blon, Blat)
        m.plot(x, y, 'r-', lw=3)
        
        mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
        mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
        etopodata = Dataset('/projects/life9360/station_map/grd_dir/etopo20.nc')
        # topoin = etopodata.variables['z'][:]
        # lons = etopodata.variables['lon'][:]
        # lats = etopodata.variables['lat'][:]
        etopo = etopodata.variables['ROSE'][:]
        lons = etopodata.variables['ETOPO20X1_1081'][:]
        lats = etopodata.variables['ETOPO20Y'][:]
        # etopo=topoin[((lats>20)*(lats<85)), :];
        # etopo=etopo[:, (lons>85)*(lons<180)];
        # lats=lats[(lats>20)*(lats<85)];
        # lons=lons[(lons>85)*(lons<180)];
        
        x, y = m(*np.meshgrid(lons,lats))
        mycm2.set_over('w',0)
        m.pcolor(x, y, etopo, cmap=mycm1, vmin=0, vmax=8000)
        m.pcolor(x, y, etopo, cmap=mycm2, vmin=-11000, vmax=-0.5)
        
        # draw parallels
        plt.show()
        return
    
    def StationDistDistribution(self, evlo, evla, mindist, maxdist, maxlat, minlat, maxlon, minlon,\
            mapflag='regional_ortho', mapfactor=4, res='i', infname=None):
        """
        Read Sation List from a txt file
        stacode longitude latidute network
        """

        LonLst1=np.array([])
        LatLst1=np.array([])
        LonLst2=np.array([])
        LatLst2=np.array([])
        for geopoint in self.geopoints:
            dist, az, baz=obsGeo.gps2DistAzimuth(evla, evlo, geopoint.lat, geopoint.lon) # distance is in m
            dist=dist/1000.
            if dist > mindist and dist < maxdist:
                LonLst1=np.append(LonLst1, geopoint.lon);
                LatLst1=np.append(LatLst1, geopoint.lat);
            else:
                LonLst2=np.append(LonLst2, geopoint.lon);
                LatLst2=np.append(LatLst2, geopoint.lat);
    
        lon_min=minlon-1
        lat_min=minlat
        lon_max=maxlon
        lat_max=maxlat
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
        if mapflag=='global':
            m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif mapflag=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
            m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/mapfactor)
            # labels = [left,right,top,bottom]
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[0,1,1,0])	
        elif mapflag=='regional_merc':
            m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1], fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[0,1,1,0], fontsize=20)
        lon = LonLst1
        lat = LatLst1
        x,y = m(lon, lat)
        m.plot(x, y, 'b^', markersize=12)
        
        # lon = LonLst2
        # lat = LatLst2
        # x,y = m(lon, lat)
        # m.plot(x, y, 'r^', markersize=5)
        evx, evy=m(evlo, evla)
        m.plot(evx, evy, 'k*', markersize=25)
        m.drawcoastlines()
        
        # url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc'
        
        # blon=np.arange(100)*(maxlon-minlon)/100.+minlon;
        # blat=np.arange(100)*(maxlat-minlat)/100.+minlat;
        # Blon=blon;
        # Blat=np.ones(Blon.size)*minlat;
        # x,y = m(Blon, Blat)
        # m.plot(x, y, 'b-', lw=3)
        
        # Blon=blon;
        # Blat=np.ones(Blon.size)*maxlat;
        # x,y = m(Blon, Blat)
        # m.plot(x, y, 'b-', lw=3)
        # 
        # Blon=np.ones(Blon.size)*minlon;
        # Blat=blat;
        # x,y = m(Blon, Blat)
        # m.plot(x, y, 'b-', lw=3)
        # 
        # Blon=np.ones(Blon.size)*maxlon;
        # Blat=blat;
        # x,y = m(Blon, Blat)
        # m.plot(x, y, 'b-', lw=3)
        
        
        if infname==None:
            mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
            mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
            etopodata = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            topoin = etopodata.variables['z'][:]
            lons = etopodata.variables['x'][:]
            lats = etopodata.variables['y'][:]
            etopo=topoin[(lats>22)*(lats<60), :];
            etopo=etopo[:, (lons>109)*(lons<163)];
            lats=lats[(lats>22)*(lats<60)];
            lons=lons[(lons>109)*(lons<163)];
            x, y = m(*np.meshgrid(lons,lats))
            mycm2.set_over('w',0)
            m.pcolor(x, y, etopo, cmap=mycm1, vmin=0, vmax=8000)
            m.pcolor(x, y, etopo, cmap=mycm2, vmin=-11000, vmax=0)
            
        else:
            # print 'Here'
            # mycm=pycpt.load.gmtColormap('/projects/life9360/code/ses3dPy/10sec/Amp.cpt')
            InArr=np.loadtxt(infname);
            lon = InArr[:,0]
            lat = InArr[:,1]
            ZValue=InArr[:,2]
            x,y = m(lon, lat)
            Nlon=int((lon.max()-lon.min())/0.5)+1
            Nlat=int((lat.max()-lat.min())/0.5)+1
            xi = np.linspace(x.min(), x.max(), Nlon)
            yi = np.linspace(y.min(), y.max(), Nlat)
            xi, yi = np.meshgrid(xi, yi)
            #-- Interpolating at the points in xi, yi
            zi = griddata(x, y, ZValue, xi, yi)
            m.pcolormesh(xi, yi, zi, cmap='gist_ncar_r', shading='gouraud', vmin=0, vmax=1100)
        plt.show()
        return
    
    
    def StationAziDistribution(self, evlo, evla, minazi, maxazi, maxlat, minlat, maxlon, minlon,\
            mapflag='regional_ortho', mapfactor=4, res='i', infname=None):
        """
        Read Sation List from a txt file
        stacode longitude latidute network
        """

        LonLst1=np.array([])
        LatLst1=np.array([])
        LonLst2=np.array([])
        LatLst2=np.array([])
        for geopoint in self.geopoints:
            dist, az, baz=obsGeo.gps2DistAzimuth(evla, evlo, geopoint.lat, geopoint.lon) # distance is in m
            dist=dist/1000.
            if az > minazi and az < maxazi:
                LonLst1=np.append(LonLst1, geopoint.lon);
                LatLst1=np.append(LatLst1, geopoint.lat);
            else:
                LonLst2=np.append(LonLst2, geopoint.lon);
                LatLst2=np.append(LatLst2, geopoint.lat);
    
        lon_min=minlon-1
        lat_min=minlat
        lon_max=maxlon
        lat_max=maxlat
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
        if mapflag=='global':
            m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif mapflag=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
            m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res, \
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/mapfactor)
            # labels = [left,right,top,bottom]
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[0,1,1,0])	
        elif mapflag=='regional_merc':
            m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
            # m = Basemap(width=6000000,height=4500000,\
            #     rsphere=(6378137.00,6356752.3142),\
            #     resolution='l',area_thresh=1000.,projection='laea',\
            #     lat_ts=40,lat_0=lat_centre,lon_0=lon_centre)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,1,1,1], fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,1,0], fontsize=20)
        lon = LonLst1
        lat = LatLst1
        x,y = m(lon, lat)
        m.plot(x, y, 'b^', markersize=12)
        # 
        # lon = LonLst2
        # lat = LatLst2
        # x,y = m(lon, lat)
        # m.plot(x, y, 'r^', markersize=5)
        evx, evy=m(evlo, evla)
        m.plot(evx, evy, 'k*', markersize=25)
        m.drawcoastlines()
        # m.etopo()
        # url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc'
        
        blon=np.arange(100)*(maxlon-minlon)/100.+minlon;
        blat=np.arange(100)*(maxlat-minlat)/100.+minlat;
        Blon=blon;
        Blat=np.ones(Blon.size)*minlat;
        x,y = m(Blon, Blat)
        m.plot(x, y, 'b-', lw=3)
        
        Blon=blon;
        Blat=np.ones(Blon.size)*maxlat;
        x,y = m(Blon, Blat)
        m.plot(x, y, 'b-', lw=3)
        
        Blon=np.ones(Blon.size)*minlon;
        Blat=blat;
        x,y = m(Blon, Blat)
        m.plot(x, y, 'b-', lw=3)
        
        Blon=np.ones(Blon.size)*maxlon;
        Blat=blat;
        x,y = m(Blon, Blat)
        m.plot(x, y, 'b-', lw=3)
        
        if infname==None:
            mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
            mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
            etopodata = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            topoin = etopodata.variables['z'][:]
            lons = etopodata.variables['x'][:]
            lats = etopodata.variables['y'][:]
            etopo=topoin[(lats>22)*(lats<60), :];
            etopo=etopo[:, (lons>109)*(lons<163)];
            lats=lats[(lats>22)*(lats<60)];
            lons=lons[(lons>109)*(lons<163)];
            x, y = m(*np.meshgrid(lons,lats))
            mycm2.set_over('w',0)
            m.pcolor(x, y, etopo, cmap=mycm1, vmin=0, vmax=8000)
            m.pcolor(x, y, etopo, cmap=mycm2, vmin=-11000, vmax=0)
            
        else:
            # print 'Here'
            # mycm=pycpt.load.gmtColormap('/projects/life9360/code/ses3dPy/10sec/Amp.cpt')
            InArr=np.loadtxt(infname);
            lon = InArr[:,0]
            lat = InArr[:,1]
            ZValue=InArr[:,2]
            # maxlon=lon.max();
            # minlon=lon.min();
            # maxlat=lat.max();
            # minlat=lat.min();
            # mylon=lon[(lon<maxlon-0.5)*(lon>minlon+0.5)*(lat<maxlat-0.5)*(lat>minlat+0.5)]
            # mylat=lat[(lon<maxlon-0.5)*(lon>minlon+0.5)*(lat<maxlat-0.5)*(lat>minlat+0.5)]
            # ZValue=ZValue[(lon<maxlon-0.5)*(lon>minlon+0.5)*(lat<maxlat-0.5)*(lat>minlat+0.5)]
            # lon=mylon;
            # lat=mylat;
            x,y = m(lon, lat)
            Nlon=int((lon.max()-lon.min())/0.5)
            Nlat=int((lat.max()-lat.min())/0.5)
            xi = np.linspace(x.min(), x.max(), Nlon)
            yi = np.linspace(y.min(), y.max(), Nlat)
            xi, yi = np.meshgrid(xi, yi)
            #-- Interpolating at the points in xi, yi
            zi = griddata(x, y, ZValue, xi, yi)
            # im=m.pcolormesh(xi, yi, zi, cmap='gist_ncar_r', shading='gouraud', vmin=0, vmax=1100)
            im=m.pcolormesh(xi, yi, zi, cmap='bwr_r', shading='gouraud', vmin=-20, vmax=20)
            cb=m.colorbar(location='bottom',size='2%')
            cb.ax.tick_params(labelsize=15)
            #cb.set_label('Phase travel time (sec)', fontsize=20)
            cb.set_label('Amplitude (nm)', fontsize=20)
            # m.colorbar(location='bottom',size='2%')
            levels=np.linspace(ZValue.min(), ZValue.max(), 40)

        plt.show()
        return 
    
    def StationSpecialDistribution(self, evlo, evla, minazi, maxazi, mindist, maxdist, maxlat, minlat, maxlon, minlon,\
            mapflag='regional_ortho', mapfactor=4, res='i', infname=None, geopolygons=[]):
        """
        Read Sation List from a txt file
        stacode longitude latidute network
        """
        ####################################################
        LonLst1=np.array([])
        LatLst1=np.array([])
        LonLst2=np.array([])
        LatLst2=np.array([])
        LonLst3=np.array([])
        LatLst3=np.array([])
        LonLst4=np.array([])
        LatLst4=np.array([])
        
        # for geopoint in self.geopoints:
        #     dist, az, baz=obsGeo.gps2DistAzimuth(evla, evlo, geopoint.lat, geopoint.lon) # distance is in m
        #     dist=dist/1000.
        #     # if az > 155 and az < 160 and dist < maxdist:
        #     if dist > 475 and dist < 525:
        #         LonLst1=np.append(LonLst1, geopoint.lon);
        #         LatLst1=np.append(LatLst1, geopoint.lat);
        #     # elif az > 250 and az < 255 and dist < maxdist:
        #     elif dist > 1025 and dist < 1075:
        #         LonLst2=np.append(LonLst2, geopoint.lon);
        #         LatLst2=np.append(LatLst2, geopoint.lat);
        #     # elif az > 270 and az < 275 and dist < maxdist:
        #     elif dist > 1675 and dist < 1725 and az<315 and az>130:
        #         LonLst3=np.append(LonLst3, geopoint.lon);
        #         LatLst3=np.append(LatLst3, geopoint.lat);
        #     else:
        #         LonLst4=np.append(LonLst4, geopoint.lon);
        #         LatLst4=np.append(LatLst4, geopoint.lat);
                
        for geopoint in self.geopoints:
            dist, az, baz=obsGeo.gps2DistAzimuth(evla, evlo, geopoint.lat, geopoint.lon) # distance is in m
            dist=dist/1000.
            if az > 235 and az < 236 :
            # if dist > 475 and dist < 525:
                LonLst1=np.append(LonLst1, geopoint.lon);
                LatLst1=np.append(LatLst1, geopoint.lat);
            elif az > 253 and az < 254 :
            # elif dist > 1025 and dist < 1075:
                LonLst2=np.append(LonLst2, geopoint.lon);
                LatLst2=np.append(LatLst2, geopoint.lat);
            elif az > 300 and az < 301 :
            # elif dist > 1675 and dist < 1725 and az<315 and az>130:
                LonLst3=np.append(LonLst3, geopoint.lon);
                LatLst3=np.append(LatLst3, geopoint.lat);
            else:
                LonLst4=np.append(LonLst4, geopoint.lon);
                LatLst4=np.append(LatLst4, geopoint.lat);
        ###################################################################################        
    
        lon_min=minlon-1
        lat_min=minlat
        lon_max=maxlon
        lat_max=maxlat
        lat_centre = (lat_max+lat_min)/2.0
        lon_centre = (lon_max+lon_min)/2.0
        fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
        if mapflag=='global':
            m = Basemap(projection='ortho', lon_0=lon_centre, lat_0=lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif mapflag=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=lon_min, lat_0=lat_min, resolution='l')
            m = Basemap(projection='ortho',lon_0=lon_min,lat_0=lat_min, resolution=res, \
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/mapfactor)
            # labels = [left,right,top,bottom]
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[0,1,1,0])	
        elif mapflag=='regional_merc':
            # m=Basemap(projection='merc',llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20,resolution=res)
            m=Basemap(projection='merc',llcrnrlat=lat_min+0.5,urcrnrlat=lat_max-1,llcrnrlon=lon_min+2,urcrnrlon=lon_max-0.5,lat_ts=20,resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,1,1,1], fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,1,0], fontsize=20)
        lon = LonLst1
        lat = LatLst1
        x,y = m(lon, lat)
        m.plot(x, y, 'b^', markersize=30)
        
        lon = LonLst2
        lat = LatLst2
        x,y = m(lon, lat)
        m.plot(x, y, 'g^', markersize=30)
        
        lon = LonLst3
        lat = LatLst3
        x,y = m(lon, lat)
        m.plot(x, y, 'r^', markersize=30)
        
        
        # lon = LonLst4
        # lat = LatLst4
        # x,y = m(lon, lat)
        # m.plot(x, y, 'g^', markersize=5)
        # evx, evy=m(evlo, evla)
        # m.plot(evx, evy, 'k*', markersize=25)
        m.drawcoastlines()
        
        # blon=np.arange(100)*(maxlon-minlon)/100.+minlon;
        # blat=np.arange(100)*(maxlat-minlat)/100.+minlat;
        # Blon=blon;
        # Blat=np.ones(Blon.size)*minlat;
        # x,y = m(Blon, Blat)
        # m.plot(x, y, 'b-', lw=3)
        # 
        # Blon=blon;
        # Blat=np.ones(Blon.size)*maxlat;
        # x,y = m(Blon, Blat)
        # m.plot(x, y, 'b-', lw=3)
        # 
        # Blon=np.ones(Blon.size)*minlon;
        # Blat=blat;
        # x,y = m(Blon, Blat)
        # m.plot(x, y, 'b-', lw=3)
        # 
        # Blon=np.ones(Blon.size)*maxlon;
        # Blat=blat;
        # x,y = m(Blon, Blat)
        # m.plot(x, y, 'b-', lw=3)
        
        if infname==None:
            mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
            mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
            etopodata = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            topoin = etopodata.variables['z'][:]
            lons = etopodata.variables['x'][:]
            lats = etopodata.variables['y'][:]
            etopo=topoin[(lats>22)*(lats<60), :];
            etopo=etopo[:, (lons>109)*(lons<163)];
            lats=lats[(lats>22)*(lats<60)];
            lons=lons[(lons>109)*(lons<163)];
            x, y = m(*np.meshgrid(lons,lats))
            mycm2.set_over('w',0)
            m.pcolor(x, y, etopo, cmap=mycm1, vmin=0, vmax=8000)
            m.pcolor(x, y, etopo, cmap=mycm2, vmin=-11000, vmax=0)
            
        else:
            # print 'Here'
            # mycm=pycpt.load.gmtColormap('/projects/life9360/code/ses3dPy/10sec/Amp.cpt')
            InArr=np.loadtxt(infname);
            lon = InArr[:,0]
            lat = InArr[:,1]
            ZValue=InArr[:,2]
            # maxlon=lon.max();
            # minlon=lon.min();
            # maxlat=lat.max();
            # minlat=lat.min();
            # mylon=lon[(lon<maxlon-0.5)*(lon>minlon+0.5)*(lat<maxlat-0.5)*(lat>minlat+0.5)]
            # mylat=lat[(lon<maxlon-0.5)*(lon>minlon+0.5)*(lat<maxlat-0.5)*(lat>minlat+0.5)]
            # ZValue=ZValue[(lon<maxlon-0.5)*(lon>minlon+0.5)*(lat<maxlat-0.5)*(lat>minlat+0.5)]
            # lon=mylon;
            # lat=mylat;
            x,y = m(lon, lat)
            Nlon=int((lon.max()-lon.min())/0.5)
            Nlat=int((lat.max()-lat.min())/0.5)
            xi = np.linspace(x.min(), x.max(), Nlon)
            yi = np.linspace(y.min(), y.max(), Nlat)
            xi, yi = np.meshgrid(xi, yi)
            #-- Interpolating at the points in xi, yi
            zi = griddata(x, y, ZValue, xi, yi)
            im=m.pcolormesh(xi, yi, zi, cmap='gist_ncar_r', shading='gouraud', vmin=0, vmax=1100)
            cb=m.colorbar(location='bottom',size='2%')
            cb.ax.tick_params(labelsize=15)
            #cb.set_label('Phase travel time (sec)', fontsize=20)
            cb.set_label('Amplitude (nm)', fontsize=20)
            # m.colorbar(location='bottom',size='2%')
            levels=np.linspace(ZValue.min(), ZValue.max(), 40)
        if len(geopolygons)!=0:
            geopolygons.PlotPolygon(mybasemap=m);
        plt.show()
        return 
    
    def Vs2VpRho(self, flag=0, Vsmin=999 ):
        for geopoint in self.geopoints:
            geopoint.Vs2VpRho(flag=flag, Vsmin= Vsmin);
        return;
        
    def GetGridGeoMap(self, maxlat, minlat, maxlon, minlon, dlon=0.5, dlat=0.5):
        LonLst=np.arange(int((maxlon-minlon)/dlon)+1)*dlon+minlon;
        LatLst=np.arange(int((maxlat-minlat)/dlat)+1)*dlat+minlat;
        L=0;
        for lon in LonLst:
            for lat in LatLst:
                tempGeoPoint=GeoPoint()
                tempGeoPoint.lon=lon;
                tempGeoPoint.lat=lat;
                tempGeoPoint.SetName();
                self.geopoints.append(tempGeoPoint);
        if (L>800):
            print 'ATTENTION: Number of receivers larger than default value!!! Check Carefully before running!';
        return;
    
    
    
def geoPEXTEND(geoP, outdir, ak135Arr, depthInter, maxdepth):
    geoP.VProfileExtend(outdir=outdir, ak135Arr=ak135Arr, depthInter=depthInter, maxdepth=maxdepth);
    return;

def PositionHExtrapolate( position, avgArr, ModelLst, outdir, geomap, Dref=500.):
    Nlon=position[0];
    Nlat=position[1];
    mindist=99999;
    mlon=0;
    mlat=0;
    # Dref=1000.;
    nameN='%g'%(Nlon) + '_%g' %(Nlat);
    if ModelLst[ModelLst==nameN].size!=0:
        return;
    print 'Extending to ', nameN
    for geopoint in geomap.geopoints:
        # print geopoint.lat, geopoint.lon, Nlat, Nlon;
        dist=great_circle((geopoint.lat, geopoint.lon),(Nlat, Nlon)).km
        # dist, az, baz=obsGeo.gps2DistAzimuth(geopoint.lat, geopoint.lon, Nlat, Nlon ) # distance is in m
        # dist=dist/1000;
        # print geopoint.lat, geopoint.lon, Nlat, Nlon;
        if dist<mindist:
            mindist=dist;
            mlon=geopoint.lon;
            mlat=geopoint.lat;
            CArr=geopoint.depthP
    newGeoP=GeoPoint();
    newGeoP.lon=Nlon;
    newGeoP.lat=Nlat;
    if mindist < Dref:
        newGeoP.depthP=((Dref-mindist)*CArr+mindist*avgArr)/Dref;
    else:
        newGeoP.depthP=avgArr;
    nameM='%g'%(mlon) + '_%g' %(mlat);
    # sfx='_'+nameM;
    sfx='_mod'
    # print 'Here to ', nameN
    newGeoP.SetVProfileFname(prefix='', suffix=sfx);
    newGeoP.SaveVProfile(outdir=outdir, dirSFX=None);
    # print 'Finished Extending to ', nameN
    return
    
class PeriodMap(object):
    """
    A class to store Phase/Group Velocity map for a specific period.
    """
    def __init__(self, period=None, mapfname='', tomomapArr=np.array([])):
        self.mapfname=mapfname;
        self.tomomapArr=tomomapArr;
        self.period=period;
        return;
        
    def ReadMap(self):
        if not os.path.isfile(self.mapfname):
            print 'Velocity Map for period: ',self.period,' sec not exist!'
            print self.mapfname;
            return;
        self.tomomapArr=np.loadtxt(self.mapfname);
        return;
    

class MapDatabase(object):
    """
    Geographical Map Database class for Map Analysis.
    """
    def __init__(self, tomodatadir='', tomotype='misha', tomof_pre='', tomof_sfx='', geo_pre='', geo_sfx='', perarray=np.array([]), geomapsdir='', refdir=''):
        self.tomodatadir=tomodatadir;
        self.tomotype=tomotype;
        self.perarray=perarray;
        self.geomapsdir=geomapsdir;
        self.refdir=refdir;
        self.tomof_pre=tomof_pre;
        self.tomof_sfx=tomof_sfx;
        
    def ReadTomoResult(self, datatype='ph'):
        """
        Read Tomoraphic Maps for a period array.
        ---------------------------------------------------------------------
        Input format:
        self.tomodatadir/per_ph/self.tomof_pre+per+self.tomof_sfx
        e.g. tomodatadir/10_ph/QC_AZI_R_1200_200_1000_100_1_10.1
        Input data are saved to a list of permap object.
        ---------------------------------------------------------------------
        """
        self.permaps=[];
        if self.tomotype=='misha':
            for per in self.perarray:
                intomofname=self.tomodatadir+'/'+'%g' %( per )+'_'+datatype+'/'+self.tomof_pre+'%g' %( per )+self.tomof_sfx;
                temp_per_map=PeriodMap(period=per, mapfname=intomofname);
                temp_per_map.ReadMap()
                self.permaps.append(temp_per_map);
        elif self.tomotype=='EH':
            for per in perarray:
                intomofname=self.tomodatadir+'/'+'%g' %( per )+'sec'+'/'+self.tomof_pre+'%g' %( per )+self.tomof_sfx;
                temp_per_map=PeriodMap(period=per, mapfname=intomofname);
                temp_per_map.ReadMap()
                self.permaps.append(temp_per_map);
        return;
    
    def TomoMap2GeoPoints(self, lonlatCheck=True, datatype='ph'):
        """
        Convert Tomographic maps to GeoMap object ( GeoPoint List ), saved as "self.geomap"
        """
        self.geomap=GeoMap();
        SizeC=self.permaps[0].tomomapArr.size;
        lonLst=self.permaps[0].tomomapArr[:,0];
        latLst=self.permaps[0].tomomapArr[:,1];
        Vvalue=self.permaps[0].tomomapArr[:,2];
        per0=self.permaps[0].period;
        for i in np.arange(lonLst.size):
            tempGeo=GeoPoint(lon=lonLst[i], lat=latLst[i]);
            tempGeo.SetName();
            if datatype=='ph':
                tempGeo.DispPh=np.array([ per0, Vvalue[i]]);
                tempGeo.SetPhDispfname(prefix='',suffix='.phv');
            elif datatype=='gr':
                tempGeo.DispGr=np.array([ per0, Vvalue[i]]);
                tempGeo.SetGrDispfname(prefix='',suffix='.grv');
            self.geomap.append(tempGeo);
        Lper=1;
        for permap in self.permaps[1:]:
            period=permap.period;
            Vvalue=permap.tomomapArr[:,2];
            if SizeC!=permap.tomomapArr.size:
                raise ValueError('Different size in period maps!: ', permap.period);
            if lonlatCheck==True:
                clon=permap.tomomapArr[:,0];
                clat=permap.tomomapArr[:,1];
                sumlon=npr.evaluate('sum(abs(lonLst-clon))');
                sumlat=npr.evaluate('sum(abs(lonLst-clon))');
                if sumlon>0.1 or sumlat>0.1:
                    raise ValueError('Incompatible grid points in period maps!: ', permap.period);
            Lper=Lper+1;
            for i in np.arange(lonLst.size):
                if datatype=='ph':
                    self.geomap[i].DispPh=np.append( self.geomap[i].DispPh , np.array( [period, Vvalue[i]]));
                elif datatype=='gr':
                    self.geomap[i].DispGr=np.append( self.geomap[i].DispGr , np.array( [period, Vvalue[i]]));
        for i in np.arange(lonLst.size):
            if datatype=='ph':
                self.geomap[i].DispPh=self.geomap[i].DispPh.reshape((Lper, 2));
            elif datatype=='gr':
                self.geomap[i].DispGr=self.geomap[i].DispGr.reshape((Lper, 2));
        return;
    
    
        
def PlotTomoMap(fname, dlon=0.5, dlat=0.5, title='', datatype='ph', outfname='', browseflag=False, saveflag=True):
    """
    Plot Tomography Map
    longitude latidute ZValue
    """
    if title=='':
        title=fname;
    if outfname=='':
        outfname=fname;
    Inarray=np.loadtxt(fname)
    LonLst=Inarray[:,0]
    LatLst=Inarray[:,1]
    ZValue=Inarray[:,2]
    llcrnrlon=LonLst.min()
    llcrnrlat=LatLst.min()
    urcrnrlon=LonLst.max()
    urcrnrlat=LatLst.max()
    Nlon=int((urcrnrlon-llcrnrlon)/dlon)+1
    Nlat=int((urcrnrlat-llcrnrlat)/dlat)+1
    fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, \
        rsphere=(6378137.00,6356752.3142), resolution='l', projection='merc')
    
    lon = LonLst
    lat = LatLst
    x,y = m(lon, lat)
    xi = np.linspace(x.min(), x.max(), Nlon)
    yi = np.linspace(y.min(), y.max(), Nlat)
    xi, yi = np.meshgrid(xi, yi)
    
    #-- Interpolating at the points in xi, yi
    zi = griddata(x, y, ZValue, xi, yi)
    # m.pcolormesh(xi, yi, zi, cmap='seismic_r', shading='gouraud')
    cmap=matplotlib.cm.seismic_r
    cmap.set_bad('w',1.)
    m.imshow(zi, cmap=cmap)
    m.drawcoastlines()
    m.colorbar(location='bottom',size='2%')
    # m.fillcontinents()
    # draw parallels
    m.drawparallels(np.arange(-90,90,10),labels=[1,1,0,1])
    # draw meridians
    m.drawmeridians(np.arange(-180,180,10),labels=[1,1,1,0])
    plt.suptitle(title,y=0.9, fontsize=22);
    if browseflag==True:
        plt.draw()
        plt.pause(1) # <-------
        raw_input("<Hit Enter To Close>")
        plt.close('all')
    if saveflag==True:
        fig.savefig(outfname+'.ps', format='ps')
    return 
    
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'
    
def PlotTomoResidualIsotropic(fname, bins=1000, xmin=-10, xmax=10, outfname='', browseflag=True, saveflag=True):

    if outfname=='':
        outfname=fname;
    Inarray=np.loadtxt(fname)
    res_tomo=Inarray[:,7];
    res_mod=Inarray[:,8];
    fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1)
    n_tomo, bins_tomo, patches_tomo = plt.hist(res_tomo, bins=bins, normed=1, facecolor='blue', alpha=0.75)
    plt.axis([xmin, xmax, 0, n_tomo.max()+0.05])
    mean_tomo=res_tomo.mean();
    std_tomo=np.std(res_tomo)
    formatter = FuncFormatter(to_percent)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('Misfit( s )', fontsize=20)
    plt.title('Tomo residual (mean: %g std: %g)' % (mean_tomo, std_tomo))             
    plt.subplot(2,1,2)
    n_mod, bins_mod, patches_mod = plt.hist(res_mod, bins=bins , normed=1, facecolor='blue', alpha=0.75)
    plt.axis([xmin, xmax, 0, n_mod.max()+0.05])
    mean_mod=res_mod.mean();
    std_mod=np.std(res_mod)
    formatter = FuncFormatter(to_percent)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('Misfit( s )', fontsize=20)
    plt.title('RefMod residual (Mean: %g std: %g)' % (mean_mod,std_mod))   
    if browseflag==True:
        plt.draw()
        plt.pause(1) # <-------
        raw_input("<Hit Enter To Close>")
        plt.close('all')
    if saveflag==True:
        fig.savefig(outfname+'.ps', format='ps')
    return
    
    
def GenerateDepthArr(depth, dz):
    if depth.size != dz.size:
        raise ValueError('Size of depth and depth interval arrays NOT compatible!');
    outArr=np.array([]);
    for i in np.arange(depth.size):
        if i==0:
            temparr=np.arange(depth[i]/dz[i])*dz[i];
        else:
            temparr=np.arange((depth[i]-depth[i-1])/dz[i])*dz[i]+depth[i-1];
        outArr=np.append(outArr, temparr);
    outArr=np.append(outArr, depth[-1]);
    return outArr;

def ak135Interpolation(ak135fname, outak135fname, refPfname):
    refPArr=np.loadtxt(refPfname);
    depthInter=refPArr[:,0];
    inak135Arr=np.loadtxt(ak135fname);
    depth=inak135Arr[:,0];
    Rou=inak135Arr[:,1];
    Vp=inak135Arr[:,2];
    Vs=inak135Arr[:,3];
    Q=inak135Arr[:,5];
    VpInter=np.interp(depthInter, depth, Vp);
    VsInter=np.interp(depthInter, depth, Vs);
    RouInter=np.interp(depthInter, depth, Rou);
    QInter=np.interp(depthInter, depth, Q);
    L=VsInter.size;
    outArr=np.append(depthInter, VsInter);
    outArr=np.append(outArr, VpInter);
    outArr=np.append(outArr, RouInter);
    outArr=np.append(outArr, QInter);
    outArr=outArr.reshape((5, L));
    outArr=outArr.T;
    np.savetxt(outak135fname, outArr, fmt='%g');
    return;


def BrocherCrustVs2VpRho(Vs):
    Vp=0.9409+2.0947*Vs-0.8206*Vs**2+0.2683*Vs**3-0.0251*Vs**4;
    Rho=1.6612*Vp-0.4721*Vp**2+0.0671*Vp**3-0.0043*Vp**4+0.000106*Vp**5;
    
    return Vp, Rho;

def MantleVs2VpRho(Vs):
    Vp=1.789*Vs;
    Rho=3.42+0.01*100*(Vs-4.5)/4.5;
    
    return Vp, Rho;

    


            
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
       
                
                
                
            
        
    
        
    
    
    

    
        
        
    
        
    
    
    
    
        
        
        