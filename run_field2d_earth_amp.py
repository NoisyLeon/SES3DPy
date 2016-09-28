import field2d_earth
import matplotlib.pyplot as plt
import GeoPolygon
basins=GeoPolygon.GeoPolygonLst()
basins.ReadGeoPolygonLst('basin1')

minlat=23.
maxlat=51.
minlon=86.
maxlon=132.

field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10., fieldtype='Amp')
Tfield=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10., fieldtype='Tph')

Tfield.read_dbase(datadir='./output_ses3d_all6')
# field.read(fname='./stf_10_20sec/Amp_10.0.txt')
field.read(fname='./stf_10sec_all/Amp_10.0.txt')
field.ZarrIn=field.ZarrIn/10.
# # field.read(fname='../Pyfmst/Tph_10sec_0.5.lst')
# field.add_noise(sigma=5.)
workingdir='./field_working'
field.interp_surface(workingdir=workingdir, outfname='Amp_10sec')
field.check_curvature(workingdir=workingdir, threshold=20.)
field.get_distArr(evlo=129.0, evla=41.306)
# field.gradient_qc(workingdir=workingdir, evlo=129.0, evla=41.306, nearneighbor=False)

field.reset_reason(10.*12+50.)

field.Laplacian_Green()
field.np2ma()
# field.plot_field(contour=False, geopolygons=basins)
# field.plot_lplc(vmin=-0.5, vmax=0.5)
# field.plot_lplcC()
field.plot_lplcC(vmin=-12, vmax=12)

# field.plot_CorrV(infield=Tfield, vmin=2.9, vmax=3.4)