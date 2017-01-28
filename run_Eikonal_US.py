import field2d_earth



minlat=24.
maxlat=50.
minlon=-120.0
maxlon=-80.
per = 10.

field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.5, minlat=minlat, maxlat=maxlat, dlat=0.5, period=per)

field.read(fname='/lustre/janus_scratch/life9360/ses3d_field_working/stf_100_10sec_US/Tph_%.1f.txt' %per)
# field.add_noise(sigma=5.)
field.interp_surface(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US', outfname='Tph_%gsec' %per)
field.check_curvature(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US')
field.gradient_qc(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US', evlo=-100., evla=40., nearneighbor=False)
field.plot_field(contour=True)
field.np2ma()
# field.plot_diffa(projection='merc')
field.plot_appV(vmin=2.8, vmax=3.4)
# field.plot_lplc()
field.write_dbase(outdir='../Pygm', gmt=True)

# field.Laplacian('convolve')
# field.plot_lplc()
# field.plot_field(contour=True)