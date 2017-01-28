#!/usr/bin/env python
import symdata
import stations
datadir = '/lustre/janus_scratch/life9360/ses3d_working_2016_US/OUTPUT'
stafile='/lustre/janus_scratch/life9360/ses3d_working_2016_US/INPUT/recfile_1'
dbase=symdata.ses3dASDF('/lustre/janus_scratch/life9360/ASDF_data/ses3d_2017_10sec_100sec_US_debug002.h5')

# # dbase.zero_padding(2.5)
evlo=-100.0
evla=40.
try:
    del dbase.events
except:
    pass
dbase.AddEvent(evlo, evla, evdp=1.0)

inftan=symdata.InputFtanParam()
inftan.ffact=1.
inftan.vmin=2.1
inftan.pmf=True
inftan.tmax=40.

try:
    del dbase.auxiliary_data.DISPbasic1
except:
    pass
try:
    del dbase.auxiliary_data.DISPbasic2
except:
    pass
try:
    del dbase.auxiliary_data.DISPpmf1
except:
    pass
try:
    del dbase.auxiliary_data.DISPpmf2
except:
    pass
# # # 
dbase.aftanMP(outdir='/lustre/janus_scratch/life9360/ses3d_working_2016_US/DISP', inftan=inftan, basic2=True,
            pmf1=True, pmf2=True, prephdir='/lustre/janus_scratch/life9360/ses3d_2017_US_phv/PRE_PHP_R', tb=0.0)

try:
    del dbase.auxiliary_data.DISPpmf2interp
except:
    pass
dbase.interpDisp(data_type='DISPpmf2')

try:
    del dbase.auxiliary_data.FieldDISPpmf2interp
except:
    pass

dbase.get_all_field(outdir='/lustre/janus_scratch/life9360/ses3d_field_working/stf_100_10sec_US_debug', data_type='DISPpmf2')

