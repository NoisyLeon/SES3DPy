#!/usr/bin/env python
import symdata
import stations
datadir = '/lustre/janus_scratch/life9360/ses3d_working_2016_US/OUTPUT'
stafile='/lustre/janus_scratch/life9360/ses3d_working_2016_US/INPUT/recfile_1'
dbase=symdata.ses3dASDF('/lustre/janus_scratch/life9360/ASDF_data/ses3d_2017_10sec_100sec_US.h5')


# # 
# # # dbase.readtxt(datadir=datadir, stafile=stafile, channel='all', verbose=True, VminPadding=2.7)
# dbase.readtxt(datadir=datadir, stafile=stafile, channel='BXZ', verbose=True, VminPadding=2.0, factor=10)
# 
# # dbase.zero_padding(2.5)
# evlo=-100.0
# evla=40.
# try:
#     del dbase.events
# except:
#     pass
# dbase.AddEvent(evlo, evla, evdp=1.0)
# # dset.pre_php(outdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016')
# # dbase.pre_php(outdir='/lustre/janus_scratch/life9360/ses3d_2016_US_phv/PRE_PHP')
# 
# # # # 
# inftan=symdata.InputFtanParam()
# inftan.ffact=1.
# inftan.vmin=2.1
# inftan.pmf=True
# # # 
# 
# # # 
# # # # try:
# # # #     del dbase.auxiliary_data.DISPpmf2interp
# # # # except:
# # # #     pass
# # # # # 
# # # # try:
# # # #     del dbase.auxiliary_data.FieldDISPpmf2interp
# # # # except:
# # # #     pass
# # # 
# try:
#     del dbase.auxiliary_data.DISPbasic1
# except:
#     pass
# try:
#     del dbase.auxiliary_data.DISPbasic2
# except:
#     pass
# try:
#     del dbase.auxiliary_data.DISPpmf1
# except:
#     pass
# try:
#     del dbase.auxiliary_data.DISPpmf2
# except:
#     pass
# # # # 
# dbase.aftanMP(outdir='/lustre/janus_scratch/life9360/ses3d_working_2016_US/DISP', inftan=inftan, basic2=True,
#             pmf1=True, pmf2=True, prephdir='/lustre/janus_scratch/life9360/ses3d_2017_US_phv/PRE_PHP_R', tb=0.0)
# # # # dbase.aftanMP(outdir='/lustre/janus_scratch/life9360/DISP_2015', inftan=inftan, basic2=True,
# # # #             pmf1=True, pmf2=True, prephdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2015_R', tb=0.0)
# # # 
# # # 
# # # # dbase.aftan(inftan=inftan, basic2=True,
# # # #             pmf1=True, pmf2=True, prephdir='/lustre/janus_scratch/life9360/PRE_PHP/ses3d_2016_R')
# # # # dbase.aftan(tb=-11.9563813705, outdir='/lustre/janus_scratch/life9360/LFMembrane_SH_0.1_20/DISP', inftan=inftan, basic2=True,
# # # #             pmf1=True, pmf2=True)
# # # 
# try:
#     del dbase.auxiliary_data.DISPpmf2interp
# except:
#     pass
# dbase.interpDisp(data_type='DISPpmf2')
# 
# try:
#     del dbase.auxiliary_data.FieldDISPpmf2interp
# except:
#     pass
# # # dbase.get_field(outdir='./stf_10sec', fieldtype='amp', data_type='DISPpmf2')
# # # dbase.get_field(outdir='./stf_10sec', fieldtype='Vgr',  data_type='DISPpmf2')
# # # dbase.get_field(outdir='./stf_10sec', fieldtype='Vph',  data_type='DISPpmf2')
# dbase.get_all_field(outdir='/lustre/janus_scratch/life9360/ses3d_field_working/stf_100_10sec_US_debug', data_type='DISPpmf2')
# # dbase.get_all_field(outdir='./stf_10_20sec', data_type='DISPpmf2')
# 
# dbase.get_field(outdir='./stf_10_20sec', data_type='DISPpmf2', fieldtype='ms')
