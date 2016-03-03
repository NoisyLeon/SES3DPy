import SES3DPy

# datadir='/lustre/janus_scratch/life9360/SES3D_WorkingDir/MODELS/MODELS_3D';
datadir='/lustre/janus_scratch/life9360/EA_model_smooth/ses3d_block_model';

mfname='dvsv'
model=SES3DPy.ses3d_model();
model.read(directory=datadir, filename=mfname)
model.smooth_horizontal(sigma=2, filter_type='neighbour')
outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/MODELS/MODELS_3D'
model.write(outdir, mfname,verbose=True)

mfname='dvsh'
model=SES3DPy.ses3d_model();
model.read(directory=datadir, filename=mfname)
model.smooth_horizontal(sigma=2, filter_type='neighbour')
outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/MODELS/MODELS_3D'
model.write(outdir, mfname,verbose=True)

mfname='dvp'
model=SES3DPy.ses3d_model();
model.read(directory=datadir, filename=mfname)
model.smooth_horizontal(sigma=2, filter_type='neighbour')
outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/MODELS/MODELS_3D'
model.write(outdir, mfname,verbose=True)

mfname='drho'
model=SES3DPy.ses3d_model();
model.read(directory=datadir, filename=mfname)
model.smooth_horizontal(sigma=2, filter_type='neighbour')
outdir='/lustre/janus_scratch/life9360/EA_ses3d_working_dir/MODELS/MODELS_3D'
model.write(outdir, mfname,verbose=True)

