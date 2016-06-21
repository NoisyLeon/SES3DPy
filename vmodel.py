import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import os
from lasif import rotations




###################################################################################################
#- rotation matrix
###################################################################################################
def rotation_matrix(n,phi):

    """ compute rotation matrix
    input: rotation angle phi [deg] and rotation vector n normalised to 1
    return: rotation matrix
    """
    phi=np.pi*phi/180.0
    A=np.array([ (n[0]*n[0],n[0]*n[1],n[0]*n[2]), (n[1]*n[0],n[1]*n[1],n[1]*n[2]), (n[2]*n[0],n[2]*n[1],n[2]*n[2])])
    B=np.eye(3)
    C=np.array([ (0.0,-n[2],n[1]), (n[2],0.0,-n[0]), (-n[1],n[0],0.0)])
    R=(1.0-np.cos(phi))*A+np.cos(phi)*B+np.sin(phi)*C
    return np.matrix(R)

###################################################################################################
#- rotate coordinates
###################################################################################################
def rotate_coordinates(n,phi,colat,lon):
    """ rotate colat and lon
    input: rotation angle phi [deg] and rotation vector n normalised to 1, original colatitude and longitude [deg]
    return: colat_new [deg], lon_new [deg]
    """
    # convert to radians
    colat=np.pi*colat/180.0
    lon=np.pi*lon/180.0
    # rotation matrix
    R=rotation_matrix(n,phi)
    # original position vector
    x=np.matrix([[np.cos(lon)*np.sin(colat)], [np.sin(lon)*np.sin(colat)], [np.cos(colat)]])
    # rotated position vector
    y=R*x
    # compute rotated colatitude and longitude
    colat_new=np.arccos(y[2])
    lon_new=np.arctan2(y[1],y[0])
    return float(180.0*colat_new/np.pi), float(180.0*lon_new/np.pi)

class cv2ses3d(object):
    """
    An object 
    """
    def __init__(self, minlat, maxlat, minlon, maxlon, depth, dlat, dlon, dz, inflag='block'):
        
        if inflag=='block':
            minlat=minlat-dlat[0]/2.
            maxlat=maxlat+dlat[0]/2.
            minlon=minlon-dlon[0]/2.
            maxlon=maxlon+dlon[0]/2.
        if depth.size!=dlat.size or dlat.size != dlon.size or dlon.size != dz.size:
            self.numsub=1
            self.depth=np.array([depth[0]])
            self.dx=np.array([dlat[0]])
            self.dy=np.array([dlon[0]])
            self.dz=np.array([dz[0]])
        else:
            self.numsub=depth.size
            self.depth=depth
            self.dx=dlat
            self.dy=dlon
            self.dz=dz
        self.xmin=90.-maxlat
        self.xmax=90.-minlat
        self.ymin= minlon
        self.ymax=maxlon
    
    def GetBlockArrLst(self):
        radius=6371
        self.xArrLst=[]
        self.yArrLst=[]
        self.xGArrLst=[]
        self.yGArrLst=[]
        self.zGArrLst=[]
        self.rGArrLst=[]
        self.zArrLst=[]
        self.rArrLst=[]
        dx=self.dx
        dy=self.dy
        dz=self.dz
        mindepth=0
        for numsub in np.arange(self.numsub):
            x0=self.xmin+dx[numsub]/2.
            y0=self.ymin+dy[numsub]/2.
            xG0=self.xmin
            yG0=self.ymin
            r0=(radius-self.depth[numsub])+dz[numsub]/2.
            z0=self.depth[numsub]-dz[numsub]/2.
            rG0=radius-self.depth[numsub]
            zG0=self.depth[numsub]
            nx=int((self.xmax-self.xmin)/dx[numsub])
            ny=int((self.ymax-self.ymin)/dx[numsub])
            nGx=int((self.xmax-self.xmin)/dx[numsub])+1
            nGy=int((self.ymax-self.ymin)/dx[numsub])+1
            nz=int((self.depth[numsub]-mindepth)/dz[numsub])
            nGz=int((self.depth[numsub]-mindepth)/dz[numsub])+1
            xArr=x0+np.arange(nx)*dx[numsub]
            yArr=y0+np.arange(ny)*dy[numsub]
            xGArr=xG0+np.arange(nGx)*dx[numsub]
            yGArr=yG0+np.arange(nGy)*dy[numsub]
            zArr=z0-np.arange(nz)*dz[numsub]
            rArr=r0+np.arange(nz)*dz[numsub]
            zGArr=zG0-np.arange(nGz)*dz[numsub]
            rGArr=rG0+np.arange(nGz)*dz[numsub]
            self.xArrLst.append(xArr)
            self.yArrLst.append(yArr)
            self.xGArrLst.append(xGArr)
            self.yGArrLst.append(yGArr)
            self.zArrLst.append(zArr)
            self.rArrLst.append(rArr)
            self.zGArrLst.append(zGArr)
            self.rGArrLst.append(rGArr)
            mindepth=self.depth[numsub]
        return
    
    def generateBlockFile(self, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        bxfname=outdir+'/block_x'
        byfname=outdir+'/block_y'
        bzfname=outdir+'/block_z'
        outbxArr=np.array([])
        outbyArr=np.array([])
        outbzArr=np.array([])
        outbxArr=np.append(outbxArr, self.numsub)
        outbyArr=np.append(outbyArr, self.numsub)
        outbzArr=np.append(outbzArr, self.numsub)
    
        for numsub in self.numsub-np.arange(self.numsub)-1:
            nGx=self.xGArrLst[numsub].size
            nGy=self.yGArrLst[numsub].size
            nGz=self.zGArrLst[numsub].size
            outbxArr=np.append(outbxArr, nGx)
            outbyArr=np.append(outbyArr, nGy)
            outbzArr=np.append(outbzArr, nGz)
            outbxArr=np.append(outbxArr, self.xGArrLst[numsub])
            outbyArr=np.append(outbyArr, self.yGArrLst[numsub])
            outbzArr=np.append(outbzArr, self.rGArrLst[numsub])
        np.savetxt(bxfname, outbxArr, fmt='%g')
        np.savetxt(byfname, outbyArr, fmt='%g')
        np.savetxt(bzfname, outbzArr, fmt='%g')
        return
    
    def generateVsLimitedGeoMap(self, datadir, outdir, Vsmin, avgfname=None, dataprx='', datasfx='_mod'):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if avgfname!=None:
            avgArr=np.loadtxt(avgfname)
            adepth=avgArr[:,0]
            arho=avgArr[:,1]
            aVp=avgArr[:,2]
            aVs=avgArr[:,3]
        depthInter=np.array([])
        NZ=np.array([0],dtype=int)
        Lnz=0
        for numsub in self.numsub-np.arange(self.numsub)-1: # maxdep ~ 0
            depthInter=np.append(depthInter, self.zArrLst[numsub])
            NZ=np.append(NZ, NZ[Lnz]+self.zArrLst[numsub].size)
            Lnz=Lnz+1
        
        depthInter=depthInter[::-1] # 0 ~ maxdep
        if avgfname!=None:
            for colat in self.xArrLst[0]:
                for lon in self.yArrLst[0]:
                    lat=90.-colat
                    infname=datadir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx
                    if not os.path.isfile(infname):
                        continue
                    inArr=np.loadtxt(infname)
                    depth=inArr[:,0]
                    Vs=inArr[:,1]
                    Vp=inArr[:,2]
                    Rho=inArr[:,3]
                    VpInter=np.interp(depthInter, depth, Vp)
                    VsInter=np.interp(depthInter, depth, Vs)
                    RhoInter=np.interp(depthInter, depth, Rho)
                    
                    aVpInter=np.interp(depthInter, adepth, aVp)
                    aVsInter=np.interp(depthInter, adepth, aVs)
                    aRhoInter=np.interp(depthInter, adepth, arho)
                    
                    if Vsmin!=999:
                        LArr=VsInter<Vsmin
                        if LArr.size!=0:
                            UArr=VsInter>=Vsmin
                            Vs1=Vsmin*LArr
                            Vs2=VsInter*UArr
                            VsInter=npr.evaluate('Vs1+Vs2')
                            # Brocher's relation
                            Vpmin=0.9409+2.0947*Vsmin-0.8206*Vsmin**2+0.2683*Vsmin**3-0.0251*Vsmin**4
                            Vp1=Vpmin*LArr
                            Vp2=VpInter*UArr
                            VpInter=npr.evaluate('Vp1+Vp2')
                            
                            Rhomin=1.6612*Vpmin-0.4721*Vpmin**2+0.0671*Vpmin**3-0.0043*Vpmin**4+0.000106*Vpmin**5
                            Rho1=Rhomin*LArr
                            Rho2=RhoInter*UArr
                            RhoInter=npr.evaluate('Rho1+Rho2')
                            
                    VpInter=npr.evaluate('VpInter-aVpInter')
                    VsInter=npr.evaluate('VsInter-aVsInter')
                    RhoInter=npr.evaluate('RhoInter-aRhoInter')
                    
                    outArr=np.append(depthInter, VsInter )
                    outArr=np.append(outArr, VpInter )
                    outArr=np.append(outArr, RhoInter )
                    outArr=outArr.reshape((4, depthInter.size))
                    outArr=outArr.T
                    outfname=outdir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx
                    np.savetxt(outfname, outArr, fmt='%g')
        else:
            for colat in self.xArrLst[0]:
                for lon in self.yArrLst[0]:
                    lat=90.-colat
                    infname=datadir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx
                    if not os.path.isfile(infname):
                        continue
                    inArr=np.loadtxt(infname)
                    depth=inArr[:,0]
                    Vs=inArr[:,1]
                    Vp=inArr[:,2]
                    Rho=inArr[:,3]
                    VpInter=np.interp(depthInter, depth, Vp)
                    VsInter=np.interp(depthInter, depth, Vs)
                    RhoInter=np.interp(depthInter, depth, Rho)
                    
                    if Vsmin!=999:
                        LArr=VsInter<Vsmin
                        if LArr.size!=0:
                            UArr=VsInter>=Vsmin
                            Vs1=Vsmin*LArr
                            Vs2=VsInter*UArr
                            VsInter=npr.evaluate('Vs1+Vs2')
                            # Brocher's relation
                            Vpmin=0.9409+2.0947*Vsmin-0.8206*Vsmin**2+0.2683*Vsmin**3-0.0251*Vsmin**4
                            Vp1=Vpmin*LArr
                            Vp2=VpInter*UArr
                            VpInter=npr.evaluate('Vp1+Vp2')
                            
                            Rhomin=1.6612*Vpmin-0.4721*Vpmin**2+0.0671*Vpmin**3-0.0043*Vpmin**4+0.000106*Vpmin**5
                            Rho1=Rhomin*LArr
                            Rho2=RhoInter*UArr
                            RhoInter=npr.evaluate('Rho1+Rho2')
                    
                    outArr=np.append(depthInter, VsInter )
                    outArr=np.append(outArr, VpInter )
                    outArr=np.append(outArr, RhoInter )
                    outArr=outArr.reshape((4, depthInter.size))
                    outArr=outArr.T
                    outfname=outdir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx
                    print outfname
                    np.savetxt(outfname,outArr,fmt='%g')
        return
    
    def generate3DModelFile(self, datadir, outdir, avgfname=None, dataprx='', datasfx='_mod', Vsmin=999):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if avgfname!=None:
            avgArr=np.loadtxt(avgfname)
            adepth=avgArr[:,0]
            arho=avgArr[:,1]
            aVp=avgArr[:,2]
            aVs=avgArr[:,3]
        
        dVpfname=outdir+'/dvp'
        dRhofname=outdir+'/drho'
        dVsvfname=outdir+'/dvsv'
        dVshfname=outdir+'/dvsh'
        
        outdVpArr=np.array([])
        outdRhoArr=np.array([])
        outdVsvArr=np.array([])
        outdVshArr=np.array([])
        
        outdVpArr=np.append(outdVpArr, self.numsub)
        outdRhoArr=np.append(outdRhoArr, self.numsub)
        outdVsvArr=np.append(outdVsvArr, self.numsub)
        outdVshArr=np.append(outdVshArr, self.numsub)
        depthInter=np.array([])
        NZ=np.array([0],dtype=int)
        Lnz=0
        for numsub in self.numsub-np.arange(self.numsub)-1: # maxdep ~ 0
            depthInter=np.append(depthInter, self.zArrLst[numsub])
            NZ=np.append(NZ, NZ[Lnz]+self.zArrLst[numsub].size)
            Lnz=Lnz+1
        
        depthInter=depthInter[::-1] # 0 ~ maxdep
        L=depthInter.size
        VpArrLst=[]
        VsArrLst=[]
        RhoArrLst=[]
        
        # dd=depthInter[::-1]
        # Lnz=0
        # for numsub in self.numsub-np.arange(self.numsub)-1: # maxdep ~ 0
        #     # print NZ[Lnz+1]
        #     print dd[NZ[Lnz]:NZ[Lnz+1]]
        #     Lnz=Lnz+1
        # return
        LH=0
        dVpmin=999
        dVsmin=999
        drhomin=999
        dVpmax=-999
        dVsmax=-999
        drhomax=-999
        if avgfname!=None:
            for colat in self.xArrLst[0]:
                for lon in self.yArrLst[0]:
                    lat=90.-colat
                    infname=datadir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx
                    inArr=np.loadtxt(infname)
                    depth=inArr[:,0]
                    Vs=inArr[:,1]
                    Vp=inArr[:,2]
                    Rho=inArr[:,3]
                    VpInter=np.interp(depthInter, depth, Vp)
                    VsInter=np.interp(depthInter, depth, Vs)
                    RhoInter=np.interp(depthInter, depth, Rho)
                    
                    aVpInter=np.interp(depthInter, adepth, aVp)
                    aVsInter=np.interp(depthInter, adepth, aVs)
                    aRhoInter=np.interp(depthInter, adepth, arho)
                    
                    if Vsmin!=999:
                        LArr=VsInter<Vsmin
                        if VsInter.min()<Vsmin:
                            UArr=VsInter>=Vsmin
                            Vs1=Vsmin*LArr
                            Vs2=VsInter*UArr
                            VsInter=npr.evaluate('Vs1+Vs2')
                            # Brocher's relation
                            Vpmin=0.9409+2.0947*Vsmin-0.8206*Vsmin**2+0.2683*Vsmin**3-0.0251*Vsmin**4
                            Vp1=Vpmin*LArr
                            Vp2=VpInter*UArr
                            VpInter=npr.evaluate('Vp1+Vp2')
                            
                            Rhomin=1.6612*Vpmin-0.4721*Vpmin**2+0.0671*Vpmin**3-0.0043*Vpmin**4+0.000106*Vpmin**5
                            Rho1=Rhomin*LArr
                            Rho2=RhoInter*UArr
                            RhoInter=npr.evaluate('Rho1+Rho2')
                            
                    VpInter=npr.evaluate('VpInter-aVpInter')
                    VsInter=npr.evaluate('VsInter-aVsInter')
                    RhoInter=npr.evaluate('RhoInter-aRhoInter')
                    
                    VpArrLst.append(VpInter[::-1])
                    VsArrLst.append(VsInter[::-1])
                    RhoArrLst.append(RhoInter[::-1]) # maxdep ~ 0
                    LH=LH+1
        else:
            for colat in self.xArrLst[0]:
                for lon in self.yArrLst[0]:
                    lat=90.-colat
                    infname=datadir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx
                    inArr=np.loadtxt(infname)
                    depth=inArr[:,0]
                    Vs=inArr[:,1]
                    Vp=inArr[:,2]
                    Rho=inArr[:,3]
                    VpInter=np.interp(depthInter, depth, Vp)
                    VsInter=np.interp(depthInter, depth, Vs)
                    RhoInter=np.interp(depthInter, depth, Rho)
                    
                    if Vsmin!=999:
                        LArr=VsInter<Vsmin
                        if VsInter.min()<Vsmin:
                            print "Revaluing: "+str(lon)+" "+str(lat)+" "+str(VsInter.min())
                            
                            UArr=VsInter>=Vsmin
                            Vs1=Vsmin*LArr
                            Vs2=VsInter*UArr
                            VsInter=npr.evaluate('Vs1+Vs2')
                            # Brocher's relation
                            Vpmin=0.9409+2.0947*Vsmin-0.8206*Vsmin**2+0.2683*Vsmin**3-0.0251*Vsmin**4
                            Vp1=Vpmin*LArr
                            Vp2=VpInter*UArr
                            VpInter=npr.evaluate('Vp1+Vp2')
                            
                            Rhomin=1.6612*Vpmin-0.4721*Vpmin**2+0.0671*Vpmin**3-0.0043*Vpmin**4+0.000106*Vpmin**5
                            Rho1=Rhomin*LArr
                            Rho2=RhoInter*UArr
                            RhoInter=npr.evaluate('Rho1+Rho2')
                    # dVpmin=min(dVpmin, VpInter.min())
                    # dVpmax=max(dVpmax, VpInter.max())
                    # dVsmin=min(dVsmin, VsInter.min())
                    # dVsmax=max(dVsmax, VsInter.max())
                    # drhomin=min(drhomin, RhoInter.min())
                    # drhomax=max(drhomax, RhoInter.max())
                    # 
                    VpArrLst.append(VpInter[::-1])
                    VsArrLst.append(VsInter[::-1])
                    RhoArrLst.append(RhoInter[::-1]) # maxdep ~ 0
                    
                    
                    # # outfname=outdir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx
                    # # outArr=np.append(depthInter, VsInter)
                    # # outArr=np.append(outArr, VpInter)
                    # # outArr=np.append(outArr, RhoInter)
                    # # outArr=outArr.reshape((4,L))
                    # # outArr=outArr.T
                    # # np.savetxt(outfname,outArr,fmt='%g') 
                    
                    LH=LH+1
        print LH
        # print 'Vp:', dVpmin, dVpmax, 'Vs:',dVsmin,dVsmax,'rho:',drhomin,drhomax
        print 'End of Reading data!'
        # return
        Lnz=0
        for numsub in self.numsub-np.arange(self.numsub)-1:
            nx=self.xArrLst[numsub].size
            ny=self.yArrLst[numsub].size
            nz=self.zArrLst[numsub].size
            nblock=nx*ny*nz
            outdVpArr=np.append(outdVpArr, nblock)
            outdRhoArr=np.append(outdRhoArr, nblock)
            outdVsvArr=np.append(outdVsvArr, nblock)
            outdVshArr=np.append(outdVshArr, nblock)
            # Lh=0
            for Lh in np.arange(LH):
            # for colat in self.xArrLst[numsub]:
            #     for lon in self.yArrLst[numsub]:
                    # dVp=VpArrLst[Lh]
                    # dVp=dVp[NZ[Lnz]:NZ[Lnz+1]]
                    # dVs=VsArrLst[Lh]
                    # dVsv=dVs[NZ[Lnz]:NZ[Lnz+1]]
                    # dVsh=dVsv
                    # dRho=RhoArrLst[Lh]
                    # dRho=dRho[NZ[Lnz]:NZ[Lnz+1]]
                    outdVpArr=np.append(outdVpArr, VpArrLst[Lh][NZ[Lnz]:NZ[Lnz+1]])
                    outdRhoArr=np.append(outdRhoArr, RhoArrLst[Lh][NZ[Lnz]:NZ[Lnz+1]])
                    outdVsvArr=np.append(outdVsvArr, VsArrLst[Lh][NZ[Lnz]:NZ[Lnz+1]])
                    outdVshArr=np.append(outdVshArr, VsArrLst[Lh][NZ[Lnz]:NZ[Lnz+1]])
                    # Lh=Lh+1
            Lnz=Lnz+1
        print 'Saving data!'
        np.savetxt(dVpfname, outdVpArr, fmt='%s')
        np.savetxt(dRhofname, outdRhoArr, fmt='%s')
        np.savetxt(dVsvfname, outdVsvArr, fmt='%s')
        np.savetxt(dVshfname, outdVshArr, fmt='%s')
        return
    
    def CheckInputModel(self,datadir, dataprx='', datasfx='_mod'):
        L=0
        Le=0
        for numsub in np.arange(self.numsub):
            xArr=self.xArrLst[numsub]
            yArr=self.yArrLst[numsub]
            for x in xArr:
                for y in yArr:
                    lat=90.-x
                    lon=y
                    # print lon, lat
                    infname=datadir+'/'+dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx
                    Le=Le+1
                    if not os.path.isfile(infname):
                        print dataprx+'%g' %(lon)+'_'+'%g' %(lat)+datasfx + ' NOT exists!' 
                        L=L+1
                        Le=Le-1
        print 'Number of lacked-data grid points: ',L, Le
        return
    

class OneDimensionalModel(object):
    """
    Simple class dealing with 1D earth models.
    """
    def __init__(self, model_name):
        """
        :param model_name: The name of the used model. Possible names are:
            'ak135-F'
        """
        if model_name.lower() == "ak135-f":
            self._read_ak135f()
        else:
            msg = "Unknown model '%s'. Possible models: %s" % (
                model_name, ", ".join(MODELS.keys()))
            raise ValueError(msg)

    def _read_ak135f(self):
        data = np.loadtxt(MODELS["ak135-f"], comments="#")

        self._depth_in_km = data[:, 0]
        self._density = data[:, 1]
        self._vp = data[:, 2]
        self._vs = data[:, 3]
        self._Q_kappa = data[:, 4]
        self._Q_mu = data[:, 5]

    def get_value(self, value_name, depth):
        """
        Returns a value at a requested depth. Currently does a simple linear
        interpolation between the two closest values.
        """
        if value_name not in EXPOSED_VALUES:
            msg = "'%s' is not a valid value name. Valid names: %s" % \
                (value_name, ", ".join(EXPOSED_VALUES))
            raise ValueError(msg)
        return np.interp(depth, self._depth_in_km,
                         getattr(self, "_" + value_name))
    
    
#########################################################################
#- define submodel model class
#########################################################################
class ses3d_submodel(object):
    """ class defining an ses3d submodel
    """
    def __init__(self):
        #- coordinate lines
        self.lat=np.zeros(1)
        self.lon=np.zeros(1)
        self.r=np.zeros(1)
        #- rotated coordinate lines
        self.lat_rot=np.zeros(1)
        self.lon_rot=np.zeros(1)
        #- field
        self.dvsv=np.zeros((1, 1, 1))
        self.dvsh=np.zeros((1, 1, 1))
        self.drho=np.zeros((1, 1, 1))
        self.dvp=np.zeros((1, 1, 1))
        
        
class ses3d_model(object):
    """
    An object for reading, writing, plotting and manipulating ses3d model
    """
    def __init__(self):
        """ initiate the ses3d_model class
        initiate list of submodels and read rotation_parameters.txt
        """
        self.nsubvol=0
        self.lat_min=0.0
        self.lat_max=0.0
        self.lon_min=0.0
        self.lon_max=0.0
        self.lat_centre=0.0
        self.lon_centre=0.0
        self.global_regional="global"
        self.m=[]
        #- read rotation parameters
        self.phi=0.0
        self.n = np.array([0., 1., 0.])
        return
    
    #########################################################################
    #- copy models
    #########################################################################
    def copy(self):
        """ Copy a model
        """
        res=ses3d_model()
        res.nsubvol=self.nsubvol
        res.lat_min=self.lat_min
        res.lat_max=self.lat_max
        res.lon_min=self.lon_min
        res.lon_max=self.lon_max
        res.lat_centre=self.lat_centre
        res.lon_centre=self.lon_centre
        res.phi=self.phi
        res.n=self.n
        res.global_regional=self.global_regional
        res.d_lon=self.d_lon
        res.d_lat=self.d_lat
        for k in np.arange(self.nsubvol):
            subvol=ses3d_submodel()
            subvol.lat=self.m[k].lat
            subvol.lon=self.m[k].lon
            subvol.r=self.m[k].r
            subvol.lat_rot=self.m[k].lat_rot
            subvol.lon_rot=self.m[k].lon_rot
            subvol.v=self.m[k].v
            res.m.append(subvol)
        return res

    #########################################################################
    #- multiplication with a scalar
    #########################################################################
    def __rmul__(self, factor):
        """ override left-multiplication of an ses3d model by a scalar factor
        """
        res=ses3d_model()
        res.nsubvol=self.nsubvol
        res.lat_min=self.lat_min
        res.lat_max=self.lat_max
        res.lon_min=self.lon_min
        res.lon_max=self.lon_max
        res.lat_centre=self.lat_centre
        res.lon_centre=self.lon_centre
        res.phi=self.phi
        res.n=self.n
        res.global_regional=self.global_regional
        res.d_lon=self.d_lon
        res.d_lat=self.d_lat
        for k in np.arange(self.nsubvol):
            subvol=ses3d_submodel()
            subvol.lat=self.m[k].lat
            subvol.lon=self.m[k].lon
            subvol.r=self.m[k].r
            subvol.lat_rot=self.m[k].lat_rot
            subvol.lon_rot=self.m[k].lon_rot
            subvol.v=factor*self.m[k].v
            res.m.append(subvol)
        return res
    
    #########################################################################
    #- adding two models
    #########################################################################
    def __add__(self,other_model):
        """ override addition of two ses3d models
        """
        res=ses3d_model()
        res.nsubvol=self.nsubvol
        res.lat_min=self.lat_min
        res.lat_max=self.lat_max
        res.lon_min=self.lon_min
        res.lon_max=self.lon_max
        res.lat_centre=self.lat_centre
        res.lon_centre=self.lon_centre
        res.phi=self.phi
        res.n=self.n
        res.global_regional=self.global_regional
        res.d_lon=self.d_lon
        res.d_lat=self.d_lat
        for k in np.arange(self.nsubvol):
            subvol=ses3d_submodel()
            subvol.lat=self.m[k].lat
            subvol.lon=self.m[k].lon
            subvol.r=self.m[k].r
            subvol.lat_rot=self.m[k].lat_rot
            subvol.lon_rot=self.m[k].lon_rot
            subvol.v=self.m[k].v+other_model.m[k].v
            res.m.append(subvol)
        return res
    
    #########################################################################
    #- read a 3D model
    #########################################################################
    def read(self, directory, filename, verbose=False):
        """ read an ses3d model from a file
        read(self, directory, filename, verbose=False):
        """
        #- read block files ====================================================
        fid_x=open(directory+'/block_x','r')
        fid_y=open(directory+'/block_y','r')
        fid_z=open(directory+'/block_z','r')
        if verbose==True:
            print 'read block files:'
            print '\t '+directory+'/block_x'
            print '\t '+directory+'/block_y'
            print '\t '+directory+'/block_z'
        ###
        # dx, dy, dz : x, y, z data in block_x/y/z
        ###
        dx=np.array(fid_x.read().strip().split('\n'),dtype=float)
        dy=np.array(fid_y.read().strip().split('\n'),dtype=float)
        dz=np.array(fid_z.read().strip().split('\n'),dtype=float)
        fid_x.close()
        fid_y.close()
        fid_z.close()
        #- read coordinate lines ===============================================
        self.nsubvol=int(dx[0])
        if verbose==True:
            print 'number of subvolumes: '+str(self.nsubvol)
        ###
        # idx, idy, idz : index for the first element in each subvolume
        ###
        idx=np.ones(self.nsubvol, dtype=int)
        idy=np.ones(self.nsubvol, dtype=int)
        idz=np.ones(self.nsubvol, dtype=int)
        for k in np.arange(1, self.nsubvol, dtype=int):
            idx[k]=int( dx[idx[k-1]] )+idx[k-1]+1
            idy[k]=int( dy[idy[k-1]] )+idy[k-1]+1
            idz[k]=int( dz[idz[k-1]] )+idz[k-1]+1
        for k in np.arange(self.nsubvol, dtype=int):
            subvol=ses3d_submodel()
            subvol.lat=90.0-dx[(idx[k]+1):(idx[k]+1+dx[idx[k]])]
            subvol.lon=dy[(idy[k]+1):(idy[k]+1+dy[idy[k]])]
            subvol.r  =dz[(idz[k]+1):(idz[k]+1+dz[idz[k]])]
            self.m.append(subvol)
        #- compute rotated version of the coordinate lines ====================
        if self.phi!=0.0:
            for k in np.arange(self.nsubvol, dtype=int):
                self.m[k].lat_rot, self.m[k].lon_rot \
                          = rotations.rotate_lat_lon(self.m[k].lat, self.m[k].lon, self.n, self.phi)
        else:
            for k in np.arange(self.nsubvol,dtype=int):
                self.m[k].lat_rot, self.m[k].lon_rot = np.meshgrid(self.m[k].lat, self.m[k].lon) ### Why meshgrid ???
                self.m[k].lat_rot = self.m[k].lat_rot.T
                self.m[k].lon_rot = self.m[k].lon_rot.T

    def read_model(self, directory, filename, verbose=False):
        #- read model volume ==================================================
        fid_m=open(directory+'/'+filename,'r')
        if verbose==True:
            print 'read model file: '+directory+'/'+filename
        v=np.array(fid_m.read().strip().split('\n'),dtype=float)
        fid_m.close()
        #- assign values ======================================================
        idx=1
        for k in np.arange(self.nsubvol):
            n = int(v[idx])
            nx = len(self.m[k].lat)-1
            ny = len(self.m[k].lon)-1
            nz = len(self.m[k].r)-1
            self.m[k].v = v[(idx+1):(idx+1+n)].reshape(nx, ny, nz)
            idx = idx+n+1
        #- decide on global or regional model==================================
        self.lat_min=90.0
        self.lat_max=-90.0
        self.lon_min=180.0
        self.lon_max=-180.0
        for k in np.arange(self.nsubvol):
            if np.min(self.m[k].lat_rot) < self.lat_min: self.lat_min = np.min(self.m[k].lat_rot)
            if np.max(self.m[k].lat_rot) > self.lat_max: self.lat_max = np.max(self.m[k].lat_rot)
            if np.min(self.m[k].lon_rot) < self.lon_min: self.lon_min = np.min(self.m[k].lon_rot)
            if np.max(self.m[k].lon_rot) > self.lon_max: self.lon_max = np.max(self.m[k].lon_rot)
        if ((self.lat_max-self.lat_min) > 90.0 or (self.lon_max-self.lon_min) > 90.0):
            self.global_regional = "global"
            self.lat_centre = (self.lat_max+self.lat_min)/2.0
            self.lon_centre = (self.lon_max+self.lon_min)/2.0
        else:
            self.global_regional = "regional"
            self.d_lat=5.0
            self.d_lon=5.0
        return
    
    #########################################################################
    #- write a 3D model to a file
    #########################################################################
    def write_model(self, directory, filename, verbose=False):
        """ write ses3d model to a file
        Output format:
        ==============================================================
        No. of subvolume
        total No. for subvolume 1
        ... (data in nx, ny, nz)
        total No. for subvolume 2
        ...
        ==============================================================
        write(self, directory, filename, verbose=False)
        """
        if not os.path.isdir(directory):
            os.makedirs(directory)
        with open(directory+'/'+filename,'w') as fid_m:
            if verbose==True:
                print 'write to file '+directory+'/'+filename
            fid_m.write(str(self.nsubvol)+'\n')
            for k in np.arange(self.nsubvol):
                nx=len(self.m[k].lat)-1
                ny=len(self.m[k].lon)-1
                nz=len(self.m[k].r)-1
                fid_m.write(str(nx*ny*nz)+'\n')
                # np.savetxt(fid_m, self.m[k].v, fmt='%g')
                for idx in np.arange(nx):
                    for idy in np.arange(ny):
                        for idz in np.arange(nz):
                            fid_m.write(str(self.m[k].v[idx,idy,idz])+'\n')
        return

    def write(self, directory, filename, verbose=False):
        """ read an ses3d model from a file
        read(self, directory, filename, verbose=False):
        """
        #- read block files ====================================================
        fid_x=open(directory+'/block_x','w')
        fid_y=open(directory+'/block_y','w')
        fid_z=open(directory+'/block_z','w')
        if verbose==True:
            print 'write block files:'
            print '\t '+directory+'/block_x'
            print '\t '+directory+'/block_y'
            print '\t '+directory+'/block_z'
        ###
        # dx, dy, dz : x, y, z data in block_x/y/z
        ###
        fid_x.write(str(self.nsubvol)+'\n')
        fid_y.write(str(self.nsubvol)+'\n')
        fid_z.write(str(self.nsubvol)+'\n')
            for k in np.arange(self.nsubvol):
                nx=len(self.m[k].lat)-1
                ny=len(self.m[k].lon)-1
                nz=len(self.m[k].r)-1
                fid_m.write(str(nx*ny*nz)+'\n')
                # np.savetxt(fid_m, self.m[k].v, fmt='%g')
                for idx in np.arange(nx):
                    for idy in np.arange(ny):
                        for idz in np.arange(nz):
                            fid_m.write(str(self.m[k].v[idx,idy,idz])+'\n')
        dx=np.array(fid_x.read().strip().split('\n'),dtype=float)
        dy=np.array(fid_y.read().strip().split('\n'),dtype=float)
        dz=np.array(fid_z.read().strip().split('\n'),dtype=float)
        fid_x.close()
        fid_y.close()
        fid_z.close()
        #- read coordinate lines ===============================================
        self.nsubvol=int(dx[0])
        if verbose==True:
            print 'number of subvolumes: '+str(self.nsubvol)
        ###
        # idx, idy, idz : index for the first element in each subvolume
        ###
        idx=np.ones(self.nsubvol, dtype=int)
        idy=np.ones(self.nsubvol, dtype=int)
        idz=np.ones(self.nsubvol, dtype=int)
        for k in np.arange(1, self.nsubvol, dtype=int):
            idx[k]=int( dx[idx[k-1]] )+idx[k-1]+1
            idy[k]=int( dy[idy[k-1]] )+idy[k-1]+1
            idz[k]=int( dz[idz[k-1]] )+idz[k-1]+1
        for k in np.arange(self.nsubvol, dtype=int):
            subvol=ses3d_submodel()
            subvol.lat=90.0-dx[(idx[k]+1):(idx[k]+1+dx[idx[k]])]
            subvol.lon=dy[(idy[k]+1):(idy[k]+1+dy[idy[k]])]
            subvol.r  =dz[(idz[k]+1):(idz[k]+1+dz[idz[k]])]
            self.m.append(subvol)
        #- compute rotated version of the coordinate lines ====================
        if self.phi!=0.0:
            for k in np.arange(self.nsubvol, dtype=int):
                self.m[k].lat_rot, self.m[k].lon_rot \
                          = rotations.rotate_lat_lon(self.m[k].lat, self.m[k].lon, self.n, self.phi)
        else:
            for k in np.arange(self.nsubvol,dtype=int):
                self.m[k].lat_rot, self.m[k].lon_rot = np.meshgrid(self.m[k].lat, self.m[k].lon) ### Why meshgrid ???
                self.m[k].lat_rot = self.m[k].lat_rot.T
                self.m[k].lon_rot = self.m[k].lon_rot.T
        #- read model volume ==================================================
        fid_m=open(directory+'/'+filename,'r')
        if verbose==True:
            print 'read model file: '+directory+'/'+filename
        v=np.array(fid_m.read().strip().split('\n'),dtype=float)
        fid_m.close()
        #- assign values ======================================================
        idx=1
        for k in np.arange(self.nsubvol):
            n = int(v[idx])
            nx = len(self.m[k].lat)-1
            ny = len(self.m[k].lon)-1
            nz = len(self.m[k].r)-1
            self.m[k].v = v[(idx+1):(idx+1+n)].reshape(nx, ny, nz)
            idx = idx+n+1
        #- decide on global or regional model==================================
        self.lat_min=90.0
        self.lat_max=-90.0
        self.lon_min=180.0
        self.lon_max=-180.0
        for k in np.arange(self.nsubvol):
            if np.min(self.m[k].lat_rot) < self.lat_min: self.lat_min = np.min(self.m[k].lat_rot)
            if np.max(self.m[k].lat_rot) > self.lat_max: self.lat_max = np.max(self.m[k].lat_rot)
            if np.min(self.m[k].lon_rot) < self.lon_min: self.lon_min = np.min(self.m[k].lon_rot)
            if np.max(self.m[k].lon_rot) > self.lon_max: self.lon_max = np.max(self.m[k].lon_rot)
        if ((self.lat_max-self.lat_min) > 90.0 or (self.lon_max-self.lon_min) > 90.0):
            self.global_regional = "global"
            self.lat_centre = (self.lat_max+self.lat_min)/2.0
            self.lon_centre = (self.lon_max+self.lon_min)/2.0
        else:
            self.global_regional = "regional"
            self.d_lat=5.0
            self.d_lon=5.0
        return
    
    #########################################################################
    #- Compute the L2 norm.
    #########################################################################
    def norm(self):
        N=0.0
        #- Loop over subvolumes. ----------------------------------------------
        for n in np.arange(self.nsubvol):
            #- Size of the array.
            nx=len(self.m[n].lat)-1
            ny=len(self.m[n].lon)-1
            nz=len(self.m[n].r)-1
            #- Compute volume elements.
            dV=np.zeros(np.shape(self.m[n].v))
            theta=(90.0-self.m[n].lat)*np.pi/180.0
            dr=self.m[n].r[1]-self.m[n].r[0]
            dphi=(self.m[n].lon[1]-self.m[n].lon[0])*np.pi/180.0
            dtheta=theta[1]-theta[0]
            for idx in np.arange(nx):
                  for idy in np.arange(ny):
                      for idz in np.arange(nz):
                          dV[idx,idy,idz]=theta[idx]*(self.m[n].r[idz])**2
            dV=dr*dtheta*dphi*dV
            #- Integrate.
            N+=np.sum(dV*(self.m[n].v)**2)
        #- Finish. ------------------------------------------------------------
        return np.sqrt(N)
    #########################################################################
    #- Remove the upper percentiles of a model.
    #########################################################################
    def clip_percentile(self, percentile):
        """
        Clip the upper percentiles of the model. Particularly useful to remove the singularities in sensitivity kernels.
        """
        #- Loop over subvolumes to find the percentile.
        percentile_list=[]
        for n in np.arange(self.nsubvol):
            percentile_list.append(np.percentile(np.abs(self.m[n].v), percentile))
        percent=np.max(percentile_list)
        #- Clip the values above the percentile.
        for n in np.arange(self.nsubvol):
            idx=np.nonzero(np.greater(np.abs(self.m[n].v),percent))
            self.m[n].v[idx]=np.sign(self.m[n].v[idx])*percent
    #########################################################################
    #- Apply horizontal smoothing.
    #########################################################################
    def smooth_horizontal(self, sigma, filter_type='gauss'):
        """
        smooth_horizontal(self,sigma,filter='gauss')
        Experimental function for smoothing in horizontal directions.
        filter_type: gauss (Gaussian smoothing), neighbour (average over neighbouring cells)
        sigma: filter width (when filter_type='gauss') or iterations (when filter_type='neighbour')
        WARNING: Currently, the smoothing only works within each subvolume. The problem 
        of smoothing across subvolumes without having excessive algorithmic complexity 
        and with fast compute times, awaits resolution ... .
        """
        #- Loop over subvolumes.---------------------------------------------------
        for n in np.arange(self.nsubvol):
            v_filtered=self.m[n].v
            #- Size of the array.
            nx=len(self.m[n].lat)-1
            ny=len(self.m[n].lon)-1
            nz=len(self.m[n].r)-1
            #- Gaussian smoothing. --------------------------------------------------
            if filter_type=='gauss':
                #- Estimate element width.
                r=np.mean(self.m[n].r)
                dx=r*np.pi*(self.m[n].lat[0]-self.m[n].lat[1])/180.0
                #- Colat and lon fields for the small Gaussian.
                dn=3*np.ceil(sigma/dx)
                nx_min=np.round(float(nx)/2.0)-dn
                nx_max=np.round(float(nx)/2.0)+dn
                ny_min=np.round(float(ny)/2.0)-dn
                ny_max=np.round(float(ny)/2.0)+dn
                lon,colat=np.meshgrid(self.m[n].lon[ny_min:ny_max],90.0-self.m[n].lat[nx_min:nx_max])
                colat=np.pi*colat/180.0
                lon=np.pi*lon/180.0
                #- Volume element.
                dy=r*np.pi*np.sin(colat)*(self.m[n].lon[1]-self.m[n].lon[0])/180.0
                dV=dx*dy
                #- Unit vector field.
                x=np.cos(lon)*np.sin(colat)
                y=np.sin(lon)*np.sin(colat)
                z=np.cos(colat)
                #- Make a Gaussian centred in the middle of the grid. -----------------
                i=np.round(float(nx)/2.0)-1
                j=np.round(float(ny)/2.0)-1
                colat_i=np.pi*(90.0-self.m[n].lat[i])/180.0
                lon_j=np.pi*self.m[n].lon[j]/180.0
                x_i=np.cos(lon_j)*np.sin(colat_i)
                y_j=np.sin(lon_j)*np.sin(colat_i)
                z_k=np.cos(colat_i)
                #- Compute the Gaussian.
                G=x*x_i+y*y_j+z*z_k
                G=G/np.max(np.abs(G))
                G=r*np.arccos(G)
                G=np.exp(-0.5*G**2/sigma**2)/(2.0*np.pi*sigma**2)
                #- Move the Gaussian across the field. --------------------------------
                for i in np.arange(dn+1,nx-dn-1):
                    for j in np.arange(dn+1,ny-dn-1):
                        for k in np.arange(nz):
                            v_filtered[i,j,k]=np.sum(self.m[n].v[i-dn:i+dn,j-dn:j+dn,k]*G*dV)
            #- Smoothing by averaging over neighbouring cells. ----------------------
            elif filter_type=='neighbour':
                for iteration in np.arange(int(sigma)):
                    for i in np.arange(1,nx-1):
                        for j in np.arange(1,ny-1):
                            v_filtered[i,j,:]=(self.m[n].v[i,j,:]+self.m[n].v[i+1,j,:]+self.m[n].v[i-1,j,:]\
                                +self.m[n].v[i,j+1,:]+self.m[n].v[i,j-1,:])/5.0
            self.m[n].v=v_filtered
        return
        
    
    #########################################################################
    #- Apply horizontal smoothing with adaptive smoothing length.
    #########################################################################
    def smooth_horizontal_adaptive(self,sigma):
        #- Find maximum smoothing length. -------------------------------------
        sigma_max=[]
        for n in np.arange(self.nsubvol):
            sigma_max.append(np.max(sigma.m[n].v))
        #- Loop over subvolumes.-----------------------------------------------
        for n in np.arange(self.nsubvol):
            #- Size of the array.
            nx=len(self.m[n].lat)-1
            ny=len(self.m[n].lon)-1
            nz=len(self.m[n].r)-1
            #- Estimate element width.
            r=np.mean(self.m[n].r)
            dx=r*np.pi*(self.m[n].lat[0]-self.m[n].lat[1])/180.0
            #- Colat and lon fields for the small Gaussian.
            dn=2*np.round(sigma_max[n]/dx)
            nx_min=np.round(float(nx)/2.0)-dn
            nx_max=np.round(float(nx)/2.0)+dn
            ny_min=np.round(float(ny)/2.0)-dn
            ny_max=np.round(float(ny)/2.0)+dn
            lon,colat=np.meshgrid(self.m[n].lon[ny_min:ny_max],90.0-self.m[n].lat[nx_min:nx_max],dtype=float)
            colat=np.pi*colat/180.0
            lon=np.pi*lon/180.0
            #- Volume element.
            dy=r*np.pi*np.sin(colat)*(self.m[n].lon[1]-self.m[n].lon[0])/180.0
            dV=dx*dy
            #- Unit vector field.
            x=np.cos(lon)*np.sin(colat)
            y=np.sin(lon)*np.sin(colat)
            z=np.cos(colat)
            #- Make a Gaussian centred in the middle of the grid. ---------------
            i=np.round(float(nx)/2.0)-1
            j=np.round(float(ny)/2.0)-1
            colat_i=np.pi*(90.0-self.m[n].lat[i])/180.0
            lon_j=np.pi*self.m[n].lon[j]/180.0
            x_i=np.cos(lon_j)*np.sin(colat_i)
            y_j=np.sin(lon_j)*np.sin(colat_i)
            z_k=np.cos(colat_i)
            #- Distance from the central point.
            G=x*x_i+y*y_j+z*z_k
            G=G/np.max(np.abs(G))
            G=r*np.arccos(G)
            #- Move the Gaussian across the field. ------------------------------
            v_filtered=self.m[n].v
            for i in np.arange(dn+1,nx-dn-1):
                for j in np.arange(dn+1,ny-dn-1):
                    for k in np.arange(nz):
                        #- Compute the actual Gaussian.
                        s=sigma.m[n].v[i,j,k]
                        if (s>0):
                            GG=np.exp(-0.5*G**2/s**2)/(2.0*np.pi*s**2)
                            #- Apply filter.
                            v_filtered[i,j,k]=np.sum(self.m[n].v[i-dn:i+dn,j-dn:j+dn,k]*GG*dV)
            self.m[n].v=v_filtered
    #########################################################################
    #- Compute relaxed velocities from velocities at 1 s reference period.
    #########################################################################
    def ref2relax(self, datadir, qmodel='cem', nrelax=3):
        """
        ref2relax(qmodel='cem', nrelax=3)
        Assuming that the current velocity model is given at the reference period 1 s, 
        ref2relax computes the relaxed velocities. They may then be written to a file.
        For this conversion, the relaxation parameters from the relax file are taken.
        Currently implemented Q models (qmodel): cem, prem, ql6 . 
        nrelax is the number of relaxation mechnisms.
        """
        #- Read the relaxation parameters from the relax file. ----------------
        tau_p=np.zeros(nrelax)
        D_p=np.zeros(nrelax)
        relaxfname=datadir+'/'+'relax'
        fid=open(relaxfname,'r')
        fid.readline()
        for n in range(nrelax):
            tau_p[n]=float(fid.readline().strip())
        fid.readline()
        for n in range(nrelax):
            D_p[n]=float(fid.readline().strip())
        fid.close()
        #- Loop over subvolumes. ----------------------------------------------
        for k in np.arange(self.nsubvol):
            nx=len(self.m[k].lat)-1
            ny=len(self.m[k].lon)-1
            nz=len(self.m[k].r)-1
            #- Loop over radius within the subvolume. ---------------------------
            for idz in np.arange(nz):
                #- Compute Q. -----------------------------------------------------
                if qmodel=='cem':
                  Q=q.q_cem(self.m[k].r[idz])
                elif qmodel=='ql6':
                  Q=q.q_ql6(self.m[k].r[idz])
                elif qmodel=='prem':
                  Q=q.q_prem(self.m[k].r[idz])
                #- Compute A and B for the reference period of 1 s. ---------------
                A=1.0
                B=0.0
                w=2.0*np.pi
                tau=1.0/Q
                for n in range(nrelax):
                    A+=tau*D_p[n]*(w**2)*(tau_p[n]**2)/(1.0+(w**2)*(tau_p[n]**2))
                    B+=tau*D_p[n]*w*tau_p[n]/(1.0+(w**2)*(tau_p[n]**2))
                conversion_factor=(A+np.sqrt(A**2+B**2))/(A**2+B**2)
                conversion_factor=np.sqrt(0.5*conversion_factor)
                #- Correct velocities. --------------------------------------------
                self.m[k].v[:,:,idz]=conversion_factor*self.m[k].v[:,:,idz]
        return
    #########################################################################
    #- convert to vtk format
    #########################################################################
    def convert_to_vtk(self, directory, filename, verbose=False):
        """ convert ses3d model to vtk format for plotting with Paraview, VisIt, ... .
        convert_to_vtk(self,directory,filename,verbose=False):
        """
        Rref=6471.
        #- preparatory steps
        nx=np.zeros(self.nsubvol, dtype=int)
        ny=np.zeros(self.nsubvol, dtype=int)
        nz=np.zeros(self.nsubvol, dtype=int)
        N=0
        for n in np.arange(self.nsubvol):
            nx[n]=len(self.m[n].lat)
            ny[n]=len(self.m[n].lon)
            nz[n]=len(self.m[n].r)
            N=N+nx[n]*ny[n]*nz[n]
        #- open file and write header
        fid=open(directory+'/'+filename,'w')
        if verbose==True:
            print 'write to file '+directory+filename
        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        #- write grid points
        fid.write('POINTS '+str(N)+' float\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing grid points for subvolume '+str(n)
            for i in np.arange(nx[n]):
                for j in np.arange(ny[n]):
                    for k in np.arange(nz[n]):
                        theta=90.0-self.m[n].lat[i]
                        phi=self.m[n].lon[j]
                        #- rotate coordinate system
                        if self.phi!=0.0:
                            theta,phi=rotate_coordinates(self.n,-self.phi,theta,phi)
                            #- transform to cartesian coordinates and write to file
                        theta=theta*np.pi/180.0
                        phi=phi*np.pi/180.0
                        r=self.m[n].r[k]/Rref ### !!!
                        x=r*np.sin(theta)*np.cos(phi)
                        y=r*np.sin(theta)*np.sin(phi)
                        z=r*np.cos(theta)
                        fid.write(str(x)+' '+str(y)+' '+str(z)+'\n')
        #- write connectivity
        n_cells=0
        for n in np.arange(self.nsubvol):
            n_cells=n_cells+(nx[n]-1)*(ny[n]-1)*(nz[n]-1)
        fid.write('\n')
        fid.write('CELLS '+str(n_cells)+' '+str(9*n_cells)+'\n')
        count=0
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing conectivity for subvolume '+str(n)
            for i in np.arange(1,nx[n]):
                for j in np.arange(1,ny[n]):
                    for k in np.arange(1,nz[n]):
                        a=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        b=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n] 
                        c=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        d=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]
                        e=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]-1
                        f=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]
                        g=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]-1
                        h=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]
                        fid.write('8 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)+' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')
            count=count+nx[n]*ny[n]*nz[n]
        #- write cell types
        fid.write('\n')
        fid.write('CELL_TYPES '+str(n_cells)+'\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing cell types for subvolume '+str(n)
            for i in np.arange(nx[n]-1):
                for j in np.arange(ny[n]-1):
                    for k in np.arange(nz[n]-1):
                        fid.write('11\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS scalars float\n')
        fid.write('LOOKUP_TABLE mytable\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing data for subvolume '+str(n)
            idx=np.arange(nx[n])
            idx[nx[n]-1]=nx[n]-2
            idy=np.arange(ny[n])
            idy[ny[n]-1]=ny[n]-2
            idz=np.arange(nz[n])
            idz[nz[n]-1]=nz[n]-2
            for i in idx:
                for j in idy:
                    for k in idz:
                        fid.write(str(self.m[n].v[i,j,k])+'\n')
        #- clean up
        fid.close()
        return
    
    
    def convert_to_vtk_depth(self, depth, directory, filename, verbose=False):
        """ convert ses3d model to vtk format for plotting with Paraview, VisIt, ... .
        convert_to_vtk(self,directory,filename,verbose=False):
        """
        Rref=6471.
        #- preparatory steps
        nx=np.zeros(self.nsubvol, dtype=int)
        ny=np.zeros(self.nsubvol, dtype=int)
        nz=np.zeros(self.nsubvol, dtype=int)
        N=0
        for n in np.arange(self.nsubvol):
            nx[n]=len(self.m[n].lat)
            ny[n]=len(self.m[n].lon)
            nz[n]=len(self.m[n].r)
            N=N+nx[n]*ny[n]*nz[n]
        #- open file and write header
        fid=open(directory+'/'+filename,'w')
        if verbose==True:
            print 'write to file '+directory+filename
        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        #- write grid points
        fid.write('POINTS '+str(N)+' float\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing grid points for subvolume '+str(n)
            for i in np.arange(nx[n]):
                for j in np.arange(ny[n]):
                    for k in np.arange(nz[n]):
                        theta=90.0-self.m[n].lat[i]
                        phi=self.m[n].lon[j]
                        #- rotate coordinate system
                        if self.phi!=0.0:
                            theta,phi=rotate_coordinates(self.n,-self.phi,theta,phi)
                            #- transform to cartesian coordinates and write to file
                        theta=theta*np.pi/180.0
                        phi=phi*np.pi/180.0
                        r=self.m[n].r[k]/Rref ### !!!
                        x=r*np.sin(theta)*np.cos(phi)
                        y=r*np.sin(theta)*np.sin(phi)
                        z=r*np.cos(theta)
                        fid.write(str(x)+' '+str(y)+' '+str(z)+'\n')
        #- write connectivity
        n_cells=0
        for n in np.arange(self.nsubvol):
            n_cells=n_cells+(nx[n]-1)*(ny[n]-1)*(nz[n]-1)
        fid.write('\n')
        fid.write('CELLS '+str(n_cells)+' '+str(9*n_cells)+'\n')
        count=0
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing conectivity for subvolume '+str(n)
            for i in np.arange(1,nx[n]):
                for j in np.arange(1,ny[n]):
                    for k in np.arange(1,nz[n]):
                        a=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        b=count+k+(j-1)*nz[n]+(i-1)*ny[n]*nz[n] 
                        c=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]-1
                        d=count+k+(j)*nz[n]+(i-1)*ny[n]*nz[n]
                        e=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]-1
                        f=count+k+(j-1)*nz[n]+(i)*ny[n]*nz[n]
                        g=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]-1
                        h=count+k+(j)*nz[n]+(i)*ny[n]*nz[n]
                        fid.write('8 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)+' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')
            count=count+nx[n]*ny[n]*nz[n]
        #- write cell types
        fid.write('\n')
        fid.write('CELL_TYPES '+str(n_cells)+'\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing cell types for subvolume '+str(n)
            for i in np.arange(nx[n]-1):
                for j in np.arange(ny[n]-1):
                    for k in np.arange(nz[n]-1):
                        fid.write('11\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS scalars float\n')
        fid.write('LOOKUP_TABLE mytable\n')
        for n in np.arange(self.nsubvol):
            if verbose==True:
                print 'writing data for subvolume '+str(n)
            idx=np.arange(nx[n])
            idx[nx[n]-1]=nx[n]-2
            idy=np.arange(ny[n])
            idy[ny[n]-1]=ny[n]-2
            idz=np.arange(nz[n])
            idz[nz[n]-1]=nz[n]-2
            for i in idx:
                for j in idy:
                    for k in idz:
                        fid.write(str(self.m[n].v[i,j,k])+'\n')
        #- clean up
        fid.close()
        return
    #########################################################################
    #- plot horizontal slices
    #########################################################################
    def plot_slice(self, depth, min_val_plot=None, max_val_plot=None, colormap='tomo', res='i', save_under=None, verbose=False, \
                   mapfactor=2, maxlon=None, minlon=None, maxlat=None, minlat=None, geopolygons=[]):
        """ plot horizontal slices through an ses3d model
        plot_slice(self,depth,colormap='tomo',res='i',save_under=None,verbose=False)
        depth=depth in km of the slice
        colormap='tomo','mono'
        res=resolution of the map, admissible values are: c, l, i, h f
        save_under=save figure as *.png with the filename "save_under". Prevents plotting of the slice.
        """
        radius=6371.0-depth
        # ax=plt.subplot(111)
        #- set up a map and colourmap -----------------------------------------
        if self.global_regional=='regional':
            m=Basemap(projection='merc',llcrnrlat=self.lat_min,urcrnrlat=self.lat_max,llcrnrlon=self.lon_min,urcrnrlon=self.lon_max,lat_ts=20,resolution=res)
            m.drawparallels(np.arange(self.lat_min,self.lat_max,self.d_lon),labels=[1,0,0,1])
            m.drawmeridians(np.arange(self.lon_min,self.lon_max,self.d_lat),labels=[1,0,0,1])
        elif self.global_regional=='global':
            self.lat_centre = (self.lat_max+self.lat_min)/2.0
            self.lon_centre = (self.lon_max+self.lon_min)/2.0
            m=Basemap(projection='ortho',lon_0=self.lon_centre, lat_0=self.lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        elif self.global_regional=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=self.lon_min, lat_0=self.lat_min, resolution='l')
            m = Basemap(projection='ortho', lon_0=self.lon_min,lat_0=self.lat_min, resolution=res,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries()
        # m.drawmapboundary(fill_color=[1.0,1.0,1.0])
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        # m.drawlsmask(land_color='0.8', ocean_color='#99ffff')
        m.drawmapboundary(fill_color="white")
        if colormap=='tomo':
            my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0],\
                0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], \
                0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif colormap=='mono':
            my_colormap=make_colormap({0.0:[1.0,1.0,1.0], 0.15:[1.0,1.0,1.0], 0.85:[0.0,0.0,0.0], 1.0:[0.0,0.0,0.0]})
        #- loop over subvolumes to collect information ------------------------
        x_list=[]
        y_list=[]
        idz_list=[]
        N_list=[]
        for k in np.arange(self.nsubvol):
            nx=len(self.m[k].lat)
            ny=len(self.m[k].lon)
            r=self.m[k].r
            #- collect subvolumes within target depth
            if (max(r)>=radius) & (min(r)<radius):
                N_list.append(k)
                r=r[0:len(r)-1]
                idz=min(np.where(min(np.abs(r-radius))==np.abs(r-radius))[0])
                if idz==len(r): idz-=idz
                idz_list.append(idz)
                if verbose==True:
                    print 'true plotting depth: '+str(6371.0-r[idz])+' km'
                x,y=m(self.m[k].lon_rot[0:nx-1,0:ny-1]+0.25,self.m[k].lat_rot[0:nx-1,0:ny-1]-0.25)
                # x,y=m(self.m[k].lon_rot[0:nx-1,0:ny-1],self.m[k].lat_rot[0:nx-1,0:ny-1])
                x_list.append(x)
                y_list.append(y)
        #- make a (hopefully) intelligent colour scale ------------------------
        if min_val_plot is None:
            if len(N_list)>0:
                #- compute some diagnostics
                min_list=[]
                max_list=[]
                percentile_list=[]
                for k in np.arange(len(N_list)):
                    min_list.append(np.min(self.m[N_list[k]].v[:,:,idz_list[k]]))
                    max_list.append(np.max(self.m[N_list[k]].v[:,:,idz_list[k]]))
                    percentile_list.append(np.percentile(np.abs(self.m[N_list[k]].v[:,:,idz_list[k]]),99.0))
                minval=np.min(min_list)
                maxval=np.max(max_list)
                percent=np.max(percentile_list)
                #- min and max roughly centred around zero
                if (minval*maxval<0.0):
                    max_val_plot=percent
                    min_val_plot=-max_val_plot
                #- min and max not centred around zero
                else:
                    max_val_plot=maxval
                    min_val_plot=minval
        if len(geopolygons)!=0:
            geopolygons.PlotPolygon(mybasemap=m)
        #- loop over subvolumes to plot ---------------------------------------
        for k in np.arange(len(N_list)):
            im=m.pcolormesh(x_list[k],y_list[k],self.m[N_list[k]].v[:,:,idz_list[k]], shading='gouraud', cmap=my_colormap,vmin=min_val_plot,vmax=max_val_plot)
            # im=m.imshow(self.m[N_list[k]].v[:,:,idz_list[k]], cmap=my_colormap,vmin=min_val_plot,vmax=max_val_plot)
          #if colormap=='mono':
            #cs=m.contour(x_list[k],y_list[k],self.m[N_list[k]].v[:,:,idz_list[k]], colors='r',linewidths=1.0)
            #plt.clabel(cs,colors='r')
        #- make a colorbar and title ------------------------------------------
        cb=m.colorbar(im,"right", size="3%", pad='2%', )
        cb.ax.tick_params(labelsize=15)
        cb.set_label('km/sec', fontsize=20, rotation=90)
        # im.ax.tick_params(labelsize=20)
        plt.title(str(depth)+' km', fontsize=30)
        
        if minlon!=None and minlat!=None and maxlon!=None and maxlat!=None:
            blon=np.arange(100)*(maxlon-minlon)/100.+minlon
            blat=np.arange(100)*(maxlat-minlat)/100.+minlat
            Blon=blon
            Blat=np.ones(Blon.size)*minlat
            x,y = m(Blon, Blat)
            m.plot(x, y, 'b-', lw=3)
            
            Blon=blon
            Blat=np.ones(Blon.size)*maxlat
            x,y = m(Blon, Blat)
            m.plot(x, y, 'b-', lw=3)
            
            Blon=np.ones(Blon.size)*minlon
            Blat=blat
            x,y = m(Blon, Blat)
            m.plot(x, y, 'b-', lw=3)
            
            Blon=np.ones(Blon.size)*maxlon
            Blat=blat
            x,y = m(Blon, Blat)
            m.plot(x, y, 'b-', lw=3)
        
        # if len(geopolygons)!=0:
        #     geopolygons.PlotPolygon(mybasemap=m)
        #- save image if wanted -----------------------------------------------
        # if save_under is None:
        #     plt.show()
        # else:
        #     plt.savefig(save_under+'.png', format='png', dpi=200)
        #     plt.close()
        return
    #########################################################################
    #- plot depth to a certain threshold value
    #########################################################################
    def plot_threshold(self, val, min_val_plot, max_val_plot, res='i', colormap='tomo', verbose=False):
        """ plot depth to a certain threshold value 'val' in an ses3d model
        plot_threshold(val,min_val_plot,max_val_plot,colormap='tomo',verbose=False):
        val=threshold value
        min_val_plot, max_val_plot=minimum and maximum values of the colour scale
        colormap='tomo','mono'
        """
        #- set up a map and colourmap
        if self.global_regional=='regional':
            m=Basemap(projection='merc',llcrnrlat=self.lat_min,urcrnrlat=self.lat_max,\
                    llcrnrlon=self.lon_min,urcrnrlon=self.lon_max,lat_ts=20,resolution=res)
            m.drawparallels(np.arange(self.lat_min,self.lat_max,self.d_lon),labels=[1,0,0,1])
            m.drawmeridians(np.arange(self.lon_min,self.lon_max,self.d_lat),labels=[1,0,0,1])
        elif self.global_regional=='global':
            m=Basemap(projection='ortho',lon_0=self.lon_centre,lat_0=self.lat_centre,resolution=res)
            m.drawparallels(np.arange(-80.0,80.0,10.0),labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0),labels=[1,0,0,1])
        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color=[1.0,1.0,1.0])
        if colormap=='tomo':
            my_colormap=make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], \
                0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92], 0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], \
                0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif colormap=='mono':
            my_colormap=make_colormap({0.0:[1.0,1.0,1.0], 0.15:[1.0,1.0,1.0], 0.85:[0.0,0.0,0.0], 1.0:[0.0,0.0,0.0]})
        #- loop over subvolumes
        for k in np.arange(self.nsubvol):
            depth=np.zeros(np.shape(self.m[k].v[:,:,0]))
            nx=len(self.m[k].lat)
            ny=len(self.m[k].lon)
            #- find depth
            r=self.m[k].r
            r=0.5*(r[0:len(r)-1]+r[1:len(r)])
            for idx in np.arange(nx-1):
                for idy in np.arange(ny-1):
                    n=self.m[k].v[idx,idy,:]>=val
                    depth[idx,idy]=6371.0-np.max(r[n])
          #- rotate coordinate system if necessary
            lon,lat=np.meshgrid(self.m[k].lon[0:ny],self.m[k].lat[0:nx])
            if self.phi!=0.0:
                lat_rot=np.zeros(np.shape(lon),dtype=float)
                lon_rot=np.zeros(np.shape(lat),dtype=float)
                for idx in np.arange(nx):
                  for idy in np.arange(ny):
                    colat=90.0-lat[idx,idy]
                    lat_rot[idx,idy],lon_rot[idx,idy]=rotate_coordinates(self.n,-self.phi,colat,lon[idx,idy])
                    lat_rot[idx,idy]=90.0-lat_rot[idx,idy]
                lon=lon_rot
                lat=lat_rot
        #- convert to map coordinates and plot
            x,y=m(lon,lat)
            im=m.pcolor(x, y, depth, cmap=my_colormap, vmin=min_val_plot, vmax=max_val_plot)
        m.colorbar(im,"right", size="3%", pad='2%')
        plt.title('depth to '+str(val)+' km/s [km]')
        plt.show()   

 
    
        