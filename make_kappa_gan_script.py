import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--reps", type=int, default=30)
parser.add_argument("--min_kappa", type=float, default=0.05)
parser.add_argument("--max_kappa", type=float, default=0.95)
parser.add_argument("--n_sigma", type=int, default=50)
args = parser.parse_args()

kappas = np.linspace(args.min_kappa, args.max_kappa, args.n_sigma)
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

kappas=({})
N_SIGMAS={}
N_REPS={}


# Execute code
python gan.py --kappa ${{kappas[$(( ($PBS_ARRAY_INDEX-1) % $N_SIGMAS ))]}} --saveloc $WORK/gan-loss-surfaces/vary_kappa_dcgan_cifar10/results_$(( ($PBS_ARRAY_INDEX-1) / $N_SIGMAS)) --datadir /tmp
"""

kappa_str = " ".join(["{:.5f}".format(s) for s in kappas])
script = template.format(len(kappas)*reps, kappa_str, len(kappas), reps)

with open("run_kappa_gan.sh", "w") as fout:
    fout.write(script)
