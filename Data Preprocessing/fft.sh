#!/bin/bash


module load singularity

singularity run --nv /projects/kumar-lab/Vinit.simg python normalised_dataanalysis_fft.py --savefile 7 --length_traj 256
