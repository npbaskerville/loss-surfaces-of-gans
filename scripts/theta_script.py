from functools import partial
import os 
import pickle as pkl
import sys

import numpy as np
from tqdm import tqdm 

sys.path.append("..")
from rmt.complexity import theta, in_integration_domain
from rmt.constants import b_constants, univariate_gaussian_variances, complexity_constant, integration_region
from rmt.coulomb import log_complexity_exponent, log_complexity_exponent_index, log_determinant
from rmt.spectrum import density, support
from rmt.empirical import sample_spectra, sample_spectrum, mc_log_determinant

import argparse
import pickle as pkl 

parser = argparse.ArgumentParser()
parser.add_argument("--p", type=int, default=3)
parser.add_argument("--q", type=int, default=3)
parser.add_argument("--sigma", type=float, default=1.)
parser.add_argument("--kappa", type=float, default=0.9)
parser.add_argument("--xmin", type=float, default=-50)
parser.add_argument("--xmax", type=float, default=500)
parser.add_argument("--x1min", type=float, default=-15)
parser.add_argument("--x1max", type=float, default=10)
parser.add_argument("--xsteps", type=int, default=200)
parser.add_argument("--x1steps", type=int, default=75)
parser.add_argument("--saveloc", type=str)


args = parser.parse_args()

p = args.p
q = args.q
kappa = args.kappa
sigma_z = args.sigma
x = np.linspace(args.xmin, args.xmax, args.xsteps)
x1 = np.linspace(args.x1min, args.x1max, args.x1steps)
x, x1 = np.meshgrid(x, x1)

b, b1 = b_constants(p, q, sigma_z, kappa)
C = complexity_constant(p, q, sigma_z, kappa)
s_sq, s1_sq = univariate_gaussian_variances(p, q, sigma_z)
exp = np.vectorize(partial(log_complexity_exponent, b=b, b1=b1, kappa=kappa, constant=C, s_sq=s_sq, s1_sq=s1_sq))

e = exp(x, x1)

uD, uG, theta_vals = theta(x, x1, np.real(e), p, q)


results = {"x": x, "x1": x1, "exp": np.real(e), "uD": uD, "uG": uG, "theta": theta_vals}
config = {"p": p, "q": q, "kappa": kappa, "sigma": sigma_z}
with open(args.saveloc, "wb") as fout:
    pkl.dump({"results": results, "config": config}, fout)
