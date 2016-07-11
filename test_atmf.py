import pyatmf

MFile = pyatmf.ATMFDataSet('../EAmodel.h5')
# MFile.readCVmodel(indir='/home/lili/EA_MODEL/ALL_MODEL', modelname='EA_model_layered',
#                 grdlst='/home/lili/EA_MODEL/EA_grid_point.lst')

# MFile.readCVmodel(indir='/projects/life9360/EA_MODEL/ALL_MODEL', modelname='EA_model_layered',
#                 grdlst='/projects/life9360/EA_MODEL/EA_grid_point.lst')
# MFile.readCVmodel(indir='/projects/life9360/EA_MODEL/ALL_MODEL', modelname='EA_model_layered',
#                 minlat = 22, Nlat=30, dlat = 0.5, minlon = 120, Nlon =30, dlon = 0.5) 
# MFile.readCVmodel(indir='/projects/life9360/EA_MODEL/ALL_MODEL', modelname='EA_model_layered')
# MFile.verticalExtend( inname='EA_model_layered', outname='EA_model_410km', dz=[0.5, 1], depthlst=[50, 200] )

# MFile.readRefmodel(infname='ak135.mod')
MFile.verticalExtend( inname='EA_model_200_ses3d', block = True, outname='EA_model_200_ses3d_block', dz=[0.5, 1], depthlst=[50, 200], maxdepth = 200.)

# MFile.verticalExtend( inname='EA_model_200_ses3d', block = True, outname='EA_model_200_ses3d_block', dz=[0.5, 1], depthlst=[50, 200], maxdepth = 200.,
#                      outdir = '/home/lili/V_extend_model_200_block')
# MFile.getavg(modelname='EA_model_200')
# MFile.verticalExtend( inname='EA_model_layered', outname='EA_model_600km', dz=[0.5, 1], depthlst=[50, 200], maxdepth = 610., 
#                 outdir = '/home/lili/V_extend_model_600' )

# MFile.horizontalExtend(modelname='EA_model_200', minlat=15., maxlat=55., dlat=0.5, minlon=75., maxlon=145., dlon=0.5)