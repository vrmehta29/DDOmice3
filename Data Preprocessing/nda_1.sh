#!/bin/bash


module load singularity

singularity run --nv /projects/kumar-lab/Vinit.simg python nda_1.py --savefile h1
