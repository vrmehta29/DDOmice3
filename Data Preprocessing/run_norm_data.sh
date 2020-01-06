#!/bin/bash


module load singularity

singularity run --nv /projects/kumar-lab/Vinit.simg python normalised_dataanalysis.py --savefile h90_ --length_traj 90
