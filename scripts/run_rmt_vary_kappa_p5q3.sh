

#!/bin/bash

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=5gb
#PBS -J 1-50

module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/loss-surfaces-of-gans/scripts


# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1

kappas=(0.05000 0.06837 0.08673 0.10510 0.12347 0.14184 0.16020 0.17857 0.19694 0.21531 0.23367 0.25204 0.27041 0.28878 0.30714 0.32551 0.34388 0.36224 0.38061 0.39898 0.41735 0.43571 0.45408 0.47245 0.49082 0.50918 0.52755 0.54592 0.56429 0.58265 0.60102 0.61939 0.63776 0.65612 0.67449 0.69286 0.71122 0.72959 0.74796 0.76633 0.78469 0.80306 0.82143 0.83980 0.85816 0.87653 0.89490 0.91327 0.93163 0.95000)


# Execute code
python rmt_kappa_exp.py --kappa ${kappas[$(( ($PBS_ARRAY_INDEX-1)  ))]} --out /work/jr19127/gan-loss-surfaces/rmt_results/vary_kappa/p5q3/results_$(( ($PBS_ARRAY_INDEX-1))) --p 5 --q 3 --sigma 1.00000
