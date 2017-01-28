import field2d_earth
import matplotlib.pyplot as plt
import GeoPolygon


minlat=24.
maxlat=50.
minlon=-120.0
maxlon=-80.
per = 10.

Afield=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.5, minlat=minlat, maxlat=maxlat, dlat=0.5, period=per, fieldtype='Amp')
Tfield=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.5, minlat=minlat, maxlat=maxlat, dlat=0.5, period=per, fieldtype='Tph')

Afield.read(fname='/lustre/janus_scratch/life9360/ses3d_field_working/stf_100_10sec_US/Amp_%.1f.txt' %per)
Afield.interp_surface(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US', outfname='Amp_%gsec' %per)
Afield.check_curvature(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US', threshold=20.)
Afield.gradient_qc(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US', evlo=-100., evla=40., nearneighbor=False)

Tfield.read(fname='/lustre/janus_scratch/life9360/ses3d_field_working/stf_100_10sec_US/Tph_%.1f.txt' %per)
Tfield.interp_surface(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US', outfname='Tph_%gsec' %per)
Tfield.check_curvature(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US')
Tfield.gradient_qc(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US', evlo=-100., evla=40., nearneighbor=False)

Afield.Laplacian()
Afield.np2ma()
Afield.plot_CorrV(infield=Tfield, vmin=2.8, vmax=3.4, outfname='../Pygm/helm_10_phv.lst')
# 
# Tfield.read_dbase(datadir='./output_ses3d_all6')
# field.read(fname='./stf_10_20sec/Amp_10.0.txt')
# A1=field.ZarrIn.max()
# field.read(fname='./stf_10sec_all/Amp_10.0.txt')
# field.ZarrIn=field.ZarrIn/field.ZarrIn.max()*A1
# # # field.read(fname='../Pyfmst/Tph_10sec_0.5.lst')
# # field.add_noise(sigma=5.)
# workingdir='./field_working'
# field.interp_surface(workingdir=workingdir, outfname='Amp_10sec')
# # field.plot_field(contour=False, geopolygons=basins)
# field.check_curvature(workingdir=workingdir, threshold=20.)
# field.get_distArr(evlo=129.0, evla=41.306)
# field.gradient_qc(workingdir=workingdir, evlo=129.0, evla=41.306, nearneighbor=False)
# # 
# field.reset_reason(10.*12+50.)
# # 
# field.Laplacian_Green()
# field.np2ma()
# # # field.plot_field(contour=False, geopolygons=basins)
# # field.plot_lplc(vmin=-0.5, vmax=0.5)
# # # field.plot_lplcC()
# field.plot_lplcC(vmin=-12, vmax=12)
# 
# field.plot_CorrV(infield=Tfield, vmin=2.9, vmax=3.4, geopolygons=basins)