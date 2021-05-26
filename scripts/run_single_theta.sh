

#!/bin/bash

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=10gb

module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/loss-surfaces-of-gans/scripts


# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1


# Execute code
python theta_script.py --xmax 1000 --xsteps 1000 --x1steps 100 --saveloc $WORK/gan-loss-surfaces/rmt_results/single_theta.pkl
