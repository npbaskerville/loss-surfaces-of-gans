import argparse
from functools import partial
import numpy as np
from tqdm import tqdm 
import pickle as pkl

from complexity import theta, in_integration_domain
from constants import b_constants, univariate_gaussian_variances, complexity_constant, integration_region
from coulomb import log_complexity_exponent, log_determinant
from spectrum import density, support

parser = argparse.ArgumentParser()

parser.add_argument("--p", type=int, default=2)
parser.add_argument("--q", type=int, default=2)
parser.add_argument("--kappa", type=float, default=0.9)

parser.add_argument("--sigma", type=float, default=1.)

parser.add_argument("--out", type=str, required=True)

args = parser.parse_args()

p = args.p
q = args.q
kappa = args.kappa
sigma_z = args.sigma 

x = np.linspace(-1000, 25, 2000)
x1 = np.linspace(-15, 40, 100)
x, x1 = np.meshgrid(x, x1)

b, b1 = b_constants(p, q, sigma_z, kappa)
C = complexity_constant(p, q, sigma_z, kappa)
s_sq, s1_sq = univariate_gaussian_variances(p, q, sigma_z)

exp = np.vectorize(partial(log_complexity_exponent, b=b, b1=b1, kappa=kappa, constant=C, s_sq=s_sq, s1_sq=s1_sq))
e = exp(x, x1)
uD, uG, T = theta(x, x1, np.real(e), p, q)
Tps = np.vstack(np.where(T > 0)).T
uzs = [uD[Tp[0], Tp[1]] for Tp in Tps]
min_uD = None
if uzs:
    min_uD =  np.min(uzs)
uzs = [uG[Tp[0], Tp[1]] for Tp in Tps]
min_uG = None
if uzs:
    min_uG =  np.min(uzs) 
uzs = [uD[Tp[0], Tp[1]] + uG[Tp[0], Tp[1]] for Tp in Tps]
min_sum = None
if uzs:
    min_sum = np.min(uzs)


with open(args.out, "wb") as fout:
    pkl.dump([sigma_z, min_uD, min_uG, min_sum], fout)

