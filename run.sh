#!/bin/bash


module load singularity

nvidia-smi -l 1 >exp10.txt&

singularity run --nv /projects/kumar-lab/Vinit_withProb.simg python miceexp2.py --hidden_layer 512,1024,1024,512 --ntrajs 100 --epochs 15  --options 4 --model test --lr -2 --init_weights xavier --tb test

