

#!/bin/bash

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:mem=5gb
#PBS -J 1-60

module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/loss-surfaces-of-gans/scripts


# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1

sigmas=(0.0000100 0.0000131 0.0000173 0.0000227 0.0000298 0.0000392 0.0000515 0.0000677 0.0000890 0.0001169 0.0001536 0.0002019 0.0002653 0.0003486 0.0004582 0.0006021 0.0007912 0.0010398 0.0013664 0.0017957 0.0023598 0.0031012 0.0040754 0.0053557 0.0070381 0.0092491 0.0121547 0.0159731 0.0209910 0.0275853 0.0362512 0.0476394 0.0626052 0.0822724 0.1081181 0.1420831 0.1867181 0.2453751 0.3224591 0.4237587 0.5568814 0.7318242 0.9617249 1.2638482 1.6608828 2.1826447 2.8683168 3.7693910 4.9535352 6.5096752 8.5546725 11.2421004 14.7737765 19.4149195 25.5140652 33.5292415 44.0623643 57.9044398 76.0949669 100.0000000)


# Execute code
python rmt_sigmaz_exp.py --sigma ${sigmas[$(( ($PBS_ARRAY_INDEX-1)  ))]} --out /work/jr19127/gan-loss-surfaces/rmt_results/vary_sigma/p7q5/results_$(( ($PBS_ARRAY_INDEX-1))) --p 7 --q 5 --kappa 0.9
