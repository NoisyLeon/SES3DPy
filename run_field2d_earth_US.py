import field2d_earth

# minlat=25.
# maxlat=50.
# minlon=235.
# maxlon=295.

minlat=24.
maxlat=50.
minlon=-120.0
maxlon=-80.


# field.read(fname='../Pyfmst/US_phV/8sec_lf')
# field.interp_surface(workingdir='./field_working_US', outfname='8sec_v')


field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.5, minlat=minlat, maxlat=maxlat, dlat=0.5, period=10.)
# field.LoadFile(fname='./stf_100_10sec_all/Tph_10.0.txt')
# field.LoadFile(fname='./stf_10sec_all/Tph_10.0.txt')
# field.interp_surface(workingdir='./field_working', outfname='Tph_10sec')

# field.read_dbase(datadir='./output')
# field.read(fname='./stf_10sec_all/Tph_10.0.txt')
field.read(fname='/lustre/janus_scratch/life9360/ses3d_field_working/stf_100_10sec_US/Tph_10.0.txt')
# field.add_noise(sigma=5.)
field.interp_surface(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US', outfname='Tph_10sec')
field.check_curvature(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US')
field.gradient_qc(workingdir='/lustre/janus_scratch/life9360/ses3d_field_working/field_working_US', evlo=-100., evla=40., nearneighbor=False)
field.plot_field(contour=True)
field.np2ma()
# field.plot_diffa(projection='merc')
field.plot_appV(vmin=2.9, vmax=3.6)
# field.plot_lplc()
# field.write_dbase(outdir='./output', gmt=True)

# field.Laplacian('convolve')
# field.plot_lplc()
# field.plot_field(contour=True)