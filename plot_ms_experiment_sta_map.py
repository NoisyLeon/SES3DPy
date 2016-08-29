import field2d_earth
import matplotlib.pyplot as plt
import GeoPolygon
basins=GeoPolygon.GeoPolygonLst()
basins.ReadGeoPolygonLst('basin1')

minlat=22.
maxlat=52.
minlon=85.
maxlon=133.

field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10., fieldtype='Amp')
field.read(fname='./stf_10_20sec/Amp_10.0.txt')
# field.ZarrIn=field.ZarrIn/10.
workingdir='./field_working'
field.interp_surface(workingdir=workingdir, outfname='Amp_10sec')
field.plot_field_sta(contour=False, geopolygons=basins, vmin=0., vmax=1600.)
