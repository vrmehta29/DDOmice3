#!/bin/bash


module load singularity

singularity run --nv /projects/kumar-lab/Vinit.simg python gen_traj.py
