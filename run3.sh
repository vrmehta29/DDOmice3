#!/bin/bash


module load singularity

nvidia-smi -l 1 >exp10.txt&

singularity run --nv /projects/kumar-lab/Vinit_withProb.simg python miceexp3.py --hidden_layer 512,1024,2056,2056,1024,512 --ntrajs 1000 --epochs 10  --options 10 --model exp17 --lr -3 --init_weights xavier --tb 45 --read_file /projects/kumar-lab/mehtav/final_data/h200__globalvvd.h5

