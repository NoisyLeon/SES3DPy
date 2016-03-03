#!/bin/bash
datadir=/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_TravelTime_MAP
perlst=(10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50)
#perlst=(10)
#perlst=(30 32 34 36 38 40 42 44 46 48 50)
#infname=slow_phase.map
infname=NKNT_Corretion.lst
minlat=25.;
maxlat=52.;
minlon=90.;
maxlon=143.;
for per in ${perlst[@]}; do
	Cdatadir=$datadir/${per}sec;
	cd $Cdatadir
	pwd
	gmtset MAP_FRAME_TYPE fancy
	REG=-R$minlon/$maxlon/$minlat/$maxlat
	surface $infname -T0.0 -Gmy.grd -I0.2 $REG
	grd2xyz my.grd $REG > ${infname}_HD
	rm my.grd
done
