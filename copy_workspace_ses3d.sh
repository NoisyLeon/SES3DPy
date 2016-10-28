#!/bin/bash

indir=$1
outdir=$2

mkdir -p $outdir/DATA/COORDINATES
mkdir -p $outdir/DATA/LOGFILES
mkdir -p $outdir/DATA/OUTPUT
mkdir -p $outdir/MODELS/MODELS
mkdir -p $outdir/MODELS/MODELS_3D
mkdir -p $outdir/MODELS_f/MODELS
mkdir -p $outdir/MODELS_f/MODELS_3D

cp -r $indir/DOC $outdir
cp -r $indir/INPUT $outdir
cp -r $indir/MAIN $outdir
cp -r $indir/MODELS/BUILD $outdir/MODELS
cp -r $indir/MODELS/MAIN $outdir/MODELS
cp -r $indir/MODELS/SOURCE $outdir/MODELS
cp -r $indir/MODELS_f/MAIN $outdir/MODELS_f
cp -r $indir/MODELS_f/SOURCE $outdir/MODELS_f
cp -r $indir/MODELS_f/MODELS_1D $outdir/MODELS_f
cp -r $indir/NOTICE $outdir
cp -r $indir/BUILD $outdir
cp -r $indir/SOURCE $outdir
cp -r $indir/SOURCE_f $outdir

