#!/bin/bash


module load singularity

singularity run --nv /projects/kumar-lab/Vinit_withProb.simg python pre_train_miceexp3.py --hidden_layer 512,512 --ntrajs 1000 --epochs 5  --options 25 --model test --lr -3 --init_weights xavier --tb 54 --read_file /projects/kumar-lab/mehtav/fft/5_fftfeat.h5 --kmeansmodel /projects/kumar-lab/mehtav/fft/5kmeans.sav --pca /projects/kumar-lab/mehtav/fft/5pca.sav

