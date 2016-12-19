import pyatmf
#############################################################
# study region
#############################################################
minlat=22.
maxlat=52.
minlon=85.
maxlon=133.

# minlat=24.
# maxlat=50.
# minlon=-125.0
# maxlon=-66.
############################################################## 
# MFile = pyatmf.ATMFDataSet('../USmodel.h5')
MFile = pyatmf.ATMFDataSet('../EAmodel.h5')

# MFile.readCVmodel(indir='/projects/life9360/US_MODEL/Vsv', modelname='US_model', header={'depth': 0, 'vs':1, 'vp':2, 'rho':3}, sfx='.mod')
# MFile.readRefmodel()
# MFile.vertical_extend( block=True, inname='US_model', outname='US_model_410km')
# MFile.getavg(modelname='US_model_410km')
# 
# MFile.horizontal_extend(modelname='US_model_410km', minlat=minlat, maxlat=maxlat, dlat=0.25, minlon=minlon, maxlon=maxlon, dlon=0.25)
MFile.read_moho('/projects/life9360/EA_MODEL/china_moho.dat', 'EA_model_210_ses3d', minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon)
# MFile.get_moho('EA_model_210_ses3d', sigma=5, minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon)

# MFile.close()