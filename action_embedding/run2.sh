#!/bin/bash


module load singularity

singularity run --nv /projects/kumar-lab/Vinit_withProb.simg   python /home/c-mehtav/DDOmice2/action_embed.py --read_file /projects/kumar-lab/mehtav/final_data/h200__globalvvd.h5 --save_dir /projects/kumar-lab/mehtav/models/ae9/5 --lr -1 --n_trajs 1000  --iterations 5000 --batch_size 128 --embed_dim 5
singularity run --nv /projects/kumar-lab/Vinit_withProb.simg   python /home/c-mehtav/DDOmice2/action_embed.py --read_file /projects/kumar-lab/mehtav/final_data/h200__globalvvd.h5 --save_dir /projects/kumar-lab/mehtav/models/ae9/30 --lr -1 --n_trajs 1000  --iterations 5000 --batch_size 128 --embed_dim 30
singularity run --nv /projects/kumar-lab/Vinit_withProb.simg   python /home/c-mehtav/DDOmice2/action_embed.py --read_file /projects/kumar-lab/mehtav/final_data/h200__globalvvd.h5 --save_dir /projects/kumar-lab/mehtav/models/ae9/2 --lr -1 --n_trajs 1000  --iterations 5000 --batch_size 128 --embed_dim 2



