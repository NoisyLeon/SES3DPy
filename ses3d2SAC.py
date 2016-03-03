import SES3DPy

# stafile='/projects/life9360/SYN_TEST_ARTIE/DATA/TOMO_1D.ak135/station.lst_ses3d';
stafile='/projects/life9360/EA_MODEL/EA_grid_point.lst';
datadir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir_001/OUTPUT';
# datadir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/OUTPUT';
outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir_001/OUTPUT_SAC_Vel_001'
minlat=0.
maxlat=4.
minlon=10.
maxlon=18.
staLst=SES3DPy.StaLst();

# staLst.GenerateReceiverLst(minlat, maxlat, minlon, maxlon, dlat=0.1, dlon=0.1, PRX='LF');
# staLst.SES3D2SAC(datadir=datadir, outdir=outdir)
staLst.ReadStaList(stafile);

NSLst=staLst.GetGridStaLst();
# datadir='/projects/life9360/code/SEM/AXISEM_v1.1/SOLVER/ak135f_mrr_5s_step/Data';
# outdir='/projects/life9360/code/SEM/AXISEM_v1.1/SOLVER/ak135f_mrr_5s_step/SAC_Data'
# staLst.AxiSEM2SAC(evla=evla, evlo=evlo, evdp=evdp, dt=0.038899550156607503, datadir=datadir, outdir=outdir, scaling=1e9)
NSLst.SES3D2SACParallel(datadir=datadir, outdir=outdir, scaling=1., integrate=False)