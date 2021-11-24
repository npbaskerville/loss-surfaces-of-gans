import argparse
from functools import partial
import os
import pickle as pkl
import sys

import numpy as np

sys.path.append("../")
from rmt.complexity import theta, in_integration_domain
from rmt.constants import b_constants, univariate_gaussian_variances, complexity_constant, integration_region
from rmt.coulomb import log_complexity_exponent_index


parser = argparse.ArgumentParser()

parser.add_argument("--kd", type=int, default=0)
parser.add_argument("--kg", type=int, default=0)
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
kd = args.kd
kg = args.kg

b, b1 = b_constants(p, q, sigma_z, kappa)
max_x = np.sqrt(2 * (1 - kappa)) * b * 1.05

x = np.linspace(-500, max_x, 200)
x1 = np.linspace(-15, 50, 100)
x, x1 = np.meshgrid(x, x1)

C = complexity_constant(p, q, sigma_z, kappa)
s_sq, s1_sq = univariate_gaussian_variances(p, q, sigma_z)
exp = np.vectorize(
    partial(log_complexity_exponent_index, b=b, b1=b1, kappa=kappa, constant=C, s_sq=s_sq, s1_sq=s1_sq, kd=kd, kg=kg))

e = exp(x, x1)

uD, uG, theta_vals = theta(x, x1, np.real(e), p, q)


the_dir = "/".join(args.out.split("/")[:-1])
os.makedirs(the_dir, exist_ok=True)
with open(args.out, "wb") as fout:
    pkl.dump([kd, kg, theta_vals, uD, uG, sigma_z, kappa], fout)
