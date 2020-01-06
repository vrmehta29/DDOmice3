#!/bin/bash


module load singularity

singularity run --nv /projects/kumar-lab/Vinit_withProb.simg python /home/c-mehtav/DDOmice2/miceexp3.py --hidden_layer 512,1024,2056,2056,1024,512 --ntrajs 200 --epochs 10  --options 5 --model test --lr -2 --init_weights xavier --tb /projects/kumar-lab/mehtav/models/tb/60 --read_file /projects/kumar-lab/mehtav/final_data/h200__globalvvd.h5
