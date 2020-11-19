import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--max_kd", type=int, default=5)
parser.add_argument("--max_kg", type=int, default=5)
parser.add_argument("--outlier_k", type=int, default=50)
parser.add_argument("--sigma", type=float, default=1.)
parser.add_argument("--p", type=int, default=2)
parser.add_argument("--q", type=int, default=2)
parser.add_argument("--kappa", type=float, default=0.9)

args = parser.parse_args()

kds = list(range(args.max_kd))
kgs = list(range(args.max_kg))

kds.append(args.outlier_k)
kgs.append(args.outlier_k)

total_ks = len(kds) * len(kgs)

template = """
#!/bin/bash

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=5gb
#PBS -J 1-{}

module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/loss-surfaces-of-gans


# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1

kds=({})
kgs=({})

# Execute code
python rmt_vary_k_theta_exp.py --kd ${{kds[$(( ($PBS_ARRAY_INDEX-1)  ))]}} --kg ${{kgs[$(( ($PBS_ARRAY_INDEX-1)  ))]}} --out $WORK/gan-loss-surfaces/rmt_results/rmt_vary_k_theta/results_$(( ($PBS_ARRAY_INDEX-1))) --p {} --q {} --kappa {:.1f} --sigma {:.7f}
"""

kds, kgs = np.meshgrid(kds, kgs)
kds = kds.ravel()
kgs = kgs.ravel()
kd_str = " ".join([str(kd) for kd in kds])
kg_str = " ".join([str(kg) for kg in kgs])

script = template.format(total_ks, kd_str, kg_str, args.p, args.q, args.kappa, args.sigma)

with open("run_rmt_vary_k_theta_exp.sh", "w") as fout:
    fout.write(script)
