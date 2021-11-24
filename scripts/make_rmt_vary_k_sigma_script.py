import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--min_sigma_mag", type=float, default=-5)
parser.add_argument("--max_sigma_mag", type=float, default=2)
parser.add_argument("--n_sigma", type=int, default=60)
parser.add_argument("--max_kd", type=int, default=30)
parser.add_argument("--max_kg", type=int, default=30)
parser.add_argument("--p", type=int, default=5)
parser.add_argument("--q", type=int, default=5)
parser.add_argument("--kappa", type=float, default=0.5)

args = parser.parse_args()

sigmas = np.logspace(args.min_sigma_mag, args.max_sigma_mag, args.n_sigma)

kds = list(range(args.max_kd))
kgs = list(range(args.max_kg))

template = """

#!/bin/bash

#SBATCH --time=5-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#PBS -J 1-{}

module load lang/cuda
module load lang/python/anaconda/pytorch

cd /user/home/jr19127/loss-surfaces-of-gans/scripts


# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1

kds=({})
kgs=({})

# Execute code
"""

python_invocation_template = """
python rmt_vary_k_theta_exp.py --sigma {:.7f} --kd ${{kds[$(( ($SLURM_ARRAY_TASK_ID-1)  ))]}} --kg ${{kgs[$(( ($SLURM_ARRAY_TASK_ID-1)  ))]}} --out {}/sigma{:.7f}/results_$(( ($SLURM_ARRAY_TASK_ID-1))) --p {} --q {} --kappa {:.1f}
"""

outdir = f"/user/work/jr19127/gan-loss-surfaces/rmt_results/vary_sigma/p{args.p}q{args.q}"
os.makedirs(outdir, exist_ok=True)

kds, kgs = np.meshgrid(kds, kgs)
kds = kds.ravel()
kgs = kgs.ravel()
kd_str = " ".join([str(kd) for kd in kds])
kg_str = " ".join([str(kg) for kg in kgs])


script_setup = template.format(len(sigmas), kd_str, kg_str)

python_invocations = [python_invocation_template.format(sigma, outdir,  args.p, args.q, args.kappa) for sigma in sigmas]

with open(f"run_rmt_vary_sigmaz_k.sh", "w") as fout:
    fout.write(script_setup + "\n")
    for python_invocation in python_invocations:
        fout.write(python_invocation + "\n")
