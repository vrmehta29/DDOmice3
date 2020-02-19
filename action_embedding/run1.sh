#!/bin/bash


module load singularity

singularity run --nv /projects/kumar-lab/Vinit_withProb.simg   python /home/c-mehtav/DDOmice2/action_embed.py --read_file /projects/kumar-lab/mehtav/final_data/h200__globalvvd.h5 --save_dir /projects/kumar-lab/mehtav/models/ae5 --lr -1 --n_trajs 1000  --iterations 50000 --batch_size 128

