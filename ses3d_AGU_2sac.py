import SES3DPy


datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/OUTPUT';
outdir='/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_SAC'
minlat=25.
maxlat=52.
minlon=90.
maxlon=143.
staLst=SES3DPy.StaLst();
staLst.GenerateReceiverLst(minlat=minlat, maxlat=maxlat,\
    minlon=minlon, maxlon=maxlon, dlat=0.25, dlon=0.25, net='EA', PRX='');
staLst.SES3D2SACParallel(datadir=datadir, outdir=outdir, VminPadding=2., delta=0.5)
# staLst.SES3D2SAC(datadir=datadir, outdir=outdir, VminPadding=2.,delta=0.5)

