import symdata
import stations
datadir = '/lustre/janus_scratch/life9360/ses3d_working_dir_2016/OUTPUT'
stafile='/lustre/janus_scratch/life9360/ses3d_working_dir_2016/INPUT/recfile_1'

dset = symdata.ses3dASDF('/lustre/janus_scratch/life9360/ASDF_data/ses3d_2016_10sec_20sec.h5')
evlo=129.029
evla=41.306
# del dset.events
dset.AddEvent(evlo, evla, evdp=1.0)
SLst=stations.StaLst()
SLst.read(stafile)
inftan=symdata.InputFtanParam()
inftan.pmf=True
inftan.vmin=2.0
# dset.get_ms(SLst, outdir='./ms_experiment', inftan=inftan, mindist=150., maxdist=3300., inaz=300.5, daz=0.5, channel='BXZ',
#                plottype='azimuth', prephdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R')
# dset.get_ms(SLst, outdir='./ms_experiment', inftan=inftan, mindist=200., maxdist=float('inf'), inaz=235.5, daz=0.5, channel='BXZ',
#                plottype='azimuth', prephdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R')
# dset.get_ms(SLst, outdir='./ms_experiment', inftan=inftan, mindist=150., maxdist=float('inf'), inaz=253.5, daz=0.5, channel='BXZ',
#                plottype='azimuth', prephdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R')

# dset.get_ms(SLst, outdir='./ms_experiment', inftan=inftan, mindist=475., maxdist=525., minaz=120., inaz=None, daz=0.5, channel='BXZ',
#                plottype='dist', prephdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R', wfactor=1.)
# dset.get_ms(SLst, outdir='./ms_experiment', inftan=inftan, mindist=1075., maxdist=1125., minaz=120., inaz=None, daz=0.5, channel='BXZ',
#                plottype='dist', prephdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R', wfactor=1.)
# dset.get_ms(SLst, outdir='./ms_experiment', inftan=inftan, mindist=1675., maxdist=1725., minaz=120., inaz=None, daz=0.5, channel='BXZ',
#                plottype='dist', prephdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R')
