#!/bin/bash
#SBATCH -J SES3DPy
#SBATCH -o SES3DPy_%j.out
#SBATCH -e SES3DPy_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --time=96:00:00
#SBATCH --mem=MaxMemPerNode

. ~/.noisepy_config
cd /projects/life9360/code/SES3DPy
python run_fields002.py
