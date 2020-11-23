import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--min_kappa", type=float, default=0.05)
parser.add_argument("--max_kappa", type=float, default=0.95)
parser.add_argument("--n_sigma", type=int, default=60)
parser.add_argument("--p", type=int, default=2)
parser.add_argument("--q", type=int, default=2)
parser.add_argument("--sigma", type=float, default=1.)

args = parser.parse_args()

kappas = np.linspace(args.min_kappa, args.max_kappa, 50)

template = """

#!/bin/bash

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=5gb
#PBS -J 1-{}

module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/loss-surfaces-of-gans


# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1

kappas=({})


# Execute code
python rmt_sigmaz_exp.py --kappa ${{kappas[$(( ($PBS_ARRAY_INDEX-1)  ))]}} --out $WORK/gan-loss-surfaces/rmt_results/vary_kappa/results_$(( ($PBS_ARRAY_INDEX-1))) --p {} --q {} --sigma {:.5f}
"""

kappa_str = " ".join(["{:.5f}".format(s) for s in kappas])
script = template.format(len(kappas), kappa_str, args.p, args.q, args.sigma)

with open("run_rmt_vary_kappa.sh", "w") as fout:
    fout.write(script)
