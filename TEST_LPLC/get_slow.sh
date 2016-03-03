#!/bin/bash
datadir=/lustre/janus_scratch/life9360/EA_postprocessing_AGU/OUTPUT_TravelTime_MAP
perlst=(10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50)
#perlst=(30 32 34 36 38 40 42 44 46 48 50)
infname=slow_azi_NKNT.phase.c.txt.HD.2.v2
slowfname=slow_phase.map

for per in ${perlst[@]}; do
	Cdatadir=$datadir/${per}sec;
	cd $Cdatadir
	pwd
	awk '{if ($3!=0) print $0}' $infname > $slowfname
done
