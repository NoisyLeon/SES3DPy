from mpl_toolkits.basemap import Basemap, shiftgrid
from lasif import ses3d_models
import numpy as np
indir = '/lustre/janus_scratch/life9360/ses3d_working_dir_2016/MODELS_f/MODELS'
rmodel = ses3d_models.RawSES3DModelHandler(directory=indir, domain={}, model_type="earth_model")

m=Basemap(projection='ortho',lon_0=109, lat_0=37, resolution='i')
# m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
# m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
# rmodel.plot_depth_slice(component="vp", depth_in_km=3., m=m, absolute_values=True)