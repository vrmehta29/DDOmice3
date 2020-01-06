#!/bin/bash


module load singularity

singularity run --nv /projects/kumar-lab/Vinit_withProb.simg python /home/c-mehtav/DDOmice2/pre_train_miceexp3_v2.py --hidden_layer 512,1024,2056,2056,1024,512 --ntrajs 5000 --epochs 5  --options 25 --model test --lr -3 --init_weights xavier --tb /projects/kumar-lab/mehtav/models/tb/61 --readfile /projects/kumar-lab/mehtav/fft/6_fftfeat.h5  --kmeansmodel /projects/kumar-lab/mehtav/fft/6gmm.sav --pca /projects/kumar-lab/mehtav/fft/6pca.sav

