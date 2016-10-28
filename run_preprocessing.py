import input_generator
import events
import stations
import Qmodel
import numpy as np

def main():
    # outdir='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT'
    outdir='/lustre/janus_scratch/life9360/ses3d_working_2016_US/INPUT'
    #########################
    # Qmodel
    # Q_model=Qmodel.Qmodel( fmin=1.0/100., fmax=1/10.)
    # Q_model.Qdiscrete()
    # # 
    # # # D=np.array([1.684, 0.838, 1.357])
    # # # tau_s=np.array([3.2, 17.692, 74.504])
    # # # Q_model.plotQdiscrete( D=D, tau_s=tau_s )
    # Q_model.plotQdiscrete()
    # Q_model.write(outdir=outdir)
    # return
    #########################
    # #################
    # stations
    SLst=stations.StaLst()
    # SLst.homo_stalst(minlat = 22.5, Nlat = 116, minlon=85.5, Nlon = 188, dlat=0.25, dlon=0.25, net='EA')
    SLst.homo_stalst(minlat = 24.5, Nlat = 100, minlon=-119.5, Nlon = 156, dlat=0.25, dlon=0.25, net='US')
    # SLst.write(outdir='/lustre/janus_scratch/life9360/ses3d_working_2016_US/INPUT')
    # return
    #################
    # events
    # evlo=129.0
    # evla=41.306
    evlo=-100.0
    evla=40.
    Mw=4.0
    evdp=1.0
    
    dt=0.05
    num_timpstep=50000
    fmin=1./100.
    fmax=1./10.
    STF=events.STF()
    # STF.RickerIntSignal(dt=dt, npts=num_timpstep, fc=0.1)
    
    STF.StepSignal(dt=dt, npts=num_timpstep)
    STF.filter('bandpass',freqmin=fmin, freqmax=fmax )
    # STF.filter('highpass', freq=fmin, corners=4, zerophase=False)
    # STF.filter('lowpass', freq=fmax, corners=4, zerophase=False)
    # STF.plotfreq()
    # stime=STF.stats.starttime
    # STF.plot(starttime=stime, endtime=stime+200., type='relative')
    ##################
    #SES3D configuration
    
    # minlat=22.
    # maxlat=52.
    # minlon=85.
    # maxlon=133.
    minlat=24.
    maxlat=50.
    minlon=-120.0
    maxlon=-80.
    
    zmin=0.
    zmax=200.
    
    nx_global=420
    ny_global=600
    nz_global=27
    
    px=10
    py=12
    pz=3
    
    inGen=input_generator.InputFileGenerator()
    inGen.set_config(num_timpstep=num_timpstep, dt=dt, nx_global=nx_global, ny_global=ny_global, nz_global=nz_global,\
        px=px, py=py, pz=pz, minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon, zmin=zmin, zmax=zmax)
    inGen.add_explosion(longitude=evlo, latitude=evla, depth=1., m0=1.45e15)
    # STF=events.STF()
    # STF.GaussianSignal(dt=dt, npts=num_timpstep, fc= 0.1)
    # inGen.get_stf(stf=STF, fmin=0.01, fmax=0.1, vmin=1.)
    inGen.get_stf(stf=STF, vmin=2.0, fmax=1./8.)
    inGen.add_stations(SLst)
    inGen.write(outdir=outdir)
    return inGen
if __name__ == "__main__":
    tvalue=main()
