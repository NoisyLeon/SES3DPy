import pyatmf
minlat=22.
maxlat=52.
minlon=85.
maxlon=133.
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
# MFile.verticalExtend( inname='EA_model_layered', block = True, outname='EA_model_210', dz=[0.5, 1], depthlst=[50, 200], maxdepth = 210.5)

# MFile.verticalExtend(inname='EA_model_layered', block = True, outname='EA_model_210', dz=[0.5, 1], depthlst=[50, 200], maxdepth = 210, 
#                      outdir = '/lustre/janus_scratch/life9360/V_extend_model_210')
# 
# MFile.getavg(modelname='EA_model_210')

# MFile.horizontalExtend(modelname='EA_model_200', minlat=15., maxlat=55., dlat=0.5, minlon=75., maxlon=145., dlon=0.5)
# MFile.horizontalExtend(modelname='EA_model_210', minlat=15., maxlat=55., dlat=0.5, minlon=75., maxlon=145., dlon=0.5)

marr=MFile.getmoho('EA_model_210_ses3d', sigma=5, minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon)