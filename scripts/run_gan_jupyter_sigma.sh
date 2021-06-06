

#!/bin/bash

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=10gb

module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/loss-surfaces-of-gans/notebooks

python gan_notebook_sigma.py 
