import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--min_sigma_mag", type=float, default=-5)
parser.add_argument("--max_sigma_mag", type=float, default=2)
parser.add_argument("--n_sigma", type=int, default=60)
parser.add_argument("--p", type=int, default=2)
parser.add_argument("--q", type=int, default=2)
parser.add_argument("--kappa", type=float, default=0.9)

args = parser.parse_args()

sigmas = np.logspace(args.min_sigma_mag, args.max_sigma_mag, args.n_sigma)

template = """

#!/bin/bash

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=5gb
#PBS -J 1-{}

module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/loss-surfaces-of-gans/scripts


# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1

sigmas=({})


# Execute code
python rmt_sigmaz_exp.py --sigma ${{sigmas[$(( ($PBS_ARRAY_INDEX-1)  ))]}} --out {}/results_$(( ($PBS_ARRAY_INDEX-1))) --p {} --q {} --kappa {:.1f}
"""
outdir = f"/work/jr19127/gan-loss-surfaces/rmt_results/vary_sigma/p{args.p}q{args.q}"
os.makedirs(outdir, exist_ok=True)

sigma_str = " ".join(["{:.7f}".format(s) for s in sigmas])
script = template.format(len(sigmas), sigma_str, outdir,  args.p, args.q, args.kappa)

with open(f"run_rmt_vary_sigmaz_p{args.p}q{args.q}.sh", "w") as fout:
    fout.write(script)

