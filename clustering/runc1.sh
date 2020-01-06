#!/bin/bash

module load singularity

singularity run --nv /projects/kumar-lab/Vinit_mp.simg python cluster1.py --readfile 7_fft.h5 --savefile 8 --ntrajs 500
