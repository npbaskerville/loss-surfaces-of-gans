import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--reps", type=int, default=30)
parser.add_argument("--min_sigma_mag", type=float, default=-5)
parser.add_argument("--max_sigma_mag", type=float, default=2)
parser.add_argument("--n_sigma", type=int, default=60)
args = parser.parse_args()

sigmas = np.logspace(args.min_sigma_mag, args.max_sigma_mag, args.n_sigma)
reps = args.reps

template = """

#!/bin/bash

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:ngpus=2:ssd=true:mem=40gb
#PBS -J 1-{}

module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/loss-surfaces-of-gans


# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1

sigmas=({})
N_SIGMAS={}
N_REPS={}


# Execute code
python gan.py --sigma ${{sigmas[$(( ($PBS_ARRAY_INDEX-1) % $N_SIGMAS ))]}} --saveloc $WORK/gan-loss-surfaces/vary_sigma_dcgan_cifar10/results_$(( ($PBS_ARRAY_INDEX-1) / $N_SIGMAS)) --datadir /tmp
"""

sigma_str = " ".join(["{:.7f}".format(s) for s in sigmas])
script = template.format(len(sigmas)*reps, sigma_str, len(sigmas), reps)

with open("run_gan.sh", "w") as fout:
    fout.write(script)
