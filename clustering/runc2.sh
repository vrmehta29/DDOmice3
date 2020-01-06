#!/bin/bash

module load singularity

singularity run --nv /projects/kumar-lab/Vinit_mp.simg python cluster2.py --readfile 7_fftfeat.h5 --savefile 7 --n_trajs 5000 --n_clusters 25
