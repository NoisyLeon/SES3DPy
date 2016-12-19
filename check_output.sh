#!/bin/bash
datadir=/lustre/janus_scratch/life9360/ses3d_working_dir_2016/OUTPUT
sfname=$datadir/vx_0_30000
#echo $sfname
while [ ! -f $sfname ]; do
	#echo "The file '$file' appeared in directory '$path' via '$action'"
	#find $datadir -type f -name 'vx*' -not -name 'vx_0_30000' -print0 | xargs -0 rm
	#find $datadir -type f -name 'vy*' -print0 | xargs -0 rm
	find $datadir -type f -name 'vx*' -not -name 'vx_0_30000' -print > log_vx
	find $datadir -type f -name 'vy*' -print > log_vy
	sleep 1000
	xargs rm < log_vx
	xargs rm < log_vy
	sleep 500
        # do something with the file
done

rm $sfname
