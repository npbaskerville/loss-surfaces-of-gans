#!/bin/bash


#PBS -l select=1:ncpus=6:ngpus=2:ssd=true:mem=40gb
#PBS -J 1-600

module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/loss-surfaces-of-gans

# Do some stuff
echo JOB ID: ${PBS_JOBID}
echo PBS ARRAY ID: ${PBS_ARRAY_INDEX}
echo Working Directory: $(pwd)

# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1

sigmas=(0.000001 0.000005 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000, 5000)
N_SIGMAS=${#sigmas[@]}
N_REPS=30


# Execute code
python gan.py --sigma ${sigmas[$(( ($PBS_ARRAY_INDEX-1) % $N_SIGMAS ))]} --saveloc $WORK/gan-loss-surfaces/vary_sigma_dcgan_cifar10/results_$(( ($PBS_ARRAY_INDEX-1) / $N_SIGMAS)) --datadir /tmp 

