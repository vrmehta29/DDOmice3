#!/bin/bash


module load singularity

singularity run --nv /projects/kumar-lab/Vinit_withProb.simg   python /home/c-mehtav/DDOmice2/ action_embed.py --read_file /projects/kumar-lab/mehtav/final_data/h200__globalvvd.h5 --save_dir /projects/kumar-lab/mehtav/models/ae3^C-lr -5 --n_trajs 90 --iterations 10000 --batch_size 128

