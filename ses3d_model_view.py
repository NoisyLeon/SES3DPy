import SES3DPy
import matplotlib.pyplot as plt
import GeoPolygon as GeoPoly

# stafile='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/station.lst_ses3d';

datadir='/lustre/janus_scratch/life9360/EA_model_smooth/ses3d_block_model_AGU'
# datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/MODELS/MODELS_3D'
mfname='dvsv'

model=SES3DPy.ses3d_model();
model.read(directory=datadir, filename=mfname)
# model.convert_to_vtk(directory=datadir, filename='model3d.vtk')
# model.global_regional='global'
model.smooth_horizontal(sigma=2, filter_type='neighbour');


# model.convert_to_vtk(directory=datadir, filename='EAmodel3D_dvs.vtk')
# model.clip_percentile(percentile=50)
model.global_regional='regional'
depth=20;
mygeopolygons=GeoPoly.GeoPolygonLst();
mygeopolygons.ReadGeoPolygonLst('basin1');
model.global_regional='regional_ortho';
# model.lon_min=90.;
# model.lat_min=25.;
model.plot_slice(depth = depth, min_val_plot=3.2, max_val_plot=3.9, mapfactor=2.75, geopolygons=mygeopolygons)
# model.plot_slice(depth = depth, min_val_plot=2.7, max_val_plot=3.5, mapfactor=2.75)
plt.title('Vs at Depth = '+str(depth)+' km', fontsize=30)
plt.show()
# model.plot_lat_slice_depth(self, component, lat, valmin, valmax, depth, iteration=0, verbose=True)
# # model.plot_threshold(val=4.0, min_val_plot=20.0, max_val_plot=100.0, colormap='tomo',verbose=False) 