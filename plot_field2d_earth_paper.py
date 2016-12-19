import field2d_earth
import GeoPolygon
basins=GeoPolygon.GeoPolygonLst()
basins.ReadGeoPolygonLst('basin1')

minlat=23.
maxlat=51.
minlon=86.
maxlon=132.

# field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)

field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10., fieldtype='Tph')
# field.read_dbase(datadir='./output')
# field.read(fname='./stf_10_20sec/Ms_10.0.txt')
# field.read(fname='./stf_10sec_all/Amp_10.0.txt')
field.read(fname='./stf_10sec_all/Tph_10.0.txt')
# field.read(fname='../Pyfmst/Tph_10sec_0.5.lst')
workingdir='./paper2016_field2d'
field.interp_surface(workingdir=workingdir, outfname='Tph_10sec')
# field.interp_surface(workingdir=workingdir, outfname='Ms_10sec')
field.get_distArr(evlo=129.0,evla=41.306)
field.check_curvature(workingdir=workingdir)
field.gradient_qc(workingdir=workingdir, evlo=129.0, evla=41.306, nearneighbor=False)
# field.reset_reason(dist=170.)
field.np2ma()
# field.plot_field(contour=True, geopolygons=basins, vmin=0, vmax=1400.)
field.plot_diffa()
# field.plot_field(contour=False, geopolygons=basins)#, vmin=0, vmax=1600.)
# field.plot_field(contour=False, geopolygons=basins, vmin=2.5, vmax=3.9)
# field.plot_appV(geopolygons=basins)
# field.plot_lplc()
# field.write_dbase(outdir='./output_ses3d_all4')
# field.get_distArr(evlo=129.0,evla=41.306)
# field.write_dbase(outdir='./output_ses3d_all6')

