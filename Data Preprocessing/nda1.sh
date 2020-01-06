#!/bin/bash
#PBS -l walltime=7:00:00
#PBS -l nodes=1:ppn=2
#PBS -l mem=16gb
#PBS -N norm_1

echo "${PBS_JOBID}"

python nda_1.py
