from functools import partial
import numpy as np
from scipy.integrate import quad

from spectrum import density, support


def log_complexity_exponent(x, x1, kappa=0.5, b=1, b1=1, s_sq=1, s1_sq=1, constant=1):
    """
    Compute the log of hte complexity exponent at x,x1.

    :param x: float; shift of the whole matrix H
    :param x1: float; shift of the upper left matrix
    :param kappa: float; ratio of upper left block to whole matrix dimension
    :param b: float; coefficient of full rank GOE
    :param b1: float; coefficient of partial rank GOE (upper left)
    :param s_sq: float; variance of the x integration variable
    :param s1_sq: float; variance of the x1 integration variable
    :param constant: float; log of the constant multiplier of the complexity
    :return: float;
    """
    return constant - (x ** 2 / (2 * s_sq) + x1 ** 2 / (2 * s1_sq)) + log_determinant(x, x1, kappa, b, b1)


def log_complexity_exponent_index(x, x1, kappa=0.5, b=1, b1=1, s_sq=1, s1_sq=1, constant=1, kd=0, kg=0):
    log_ce = log_complexity_exponent(x, x1, kappa=kappa, b=b, b1=b1, s_sq=s_sq, s1_sq=s1_sq, constant=constant)
    if kg > 0:
        log_ce -= kg * kappa * _goe_ldp_rate(x, np.sqrt(2 * (1 - kappa)) * b)
    if kd > 0:
        log_ce -= kd * (1 - kappa) * _goe_ldp_rate(-(x + x1), np.sqrt(2 * kappa * (b ** 2 + b1 ** 2)))
    return log_ce


def log_determinant(x, x1, kappa=0.5, b=1, b1=1):
    """
    Compute the log of the Hessian H determinant.

    :param x: float; shift of the whole matrix H
    :param x1: float; shift of the upper left matrix
    :param kappa: float; ratio of upper left block to whole matrix dimension
    :param b: float; coefficient of full rank GOE
    :param b1: float; coefficient of partial rank GOE (upper left)
    :return: float;
    """
    lsd = partial(density, x1=x1, kappa=kappa, b=b, b1=b1)

    def quad_func(z):
        return np.log(np.abs(z - x)) * lsd(z)

    limits = support(x1, b, b1)
    return quad(quad_func, limits[0], limits[1])[0]


def _goe_ldp_rate(u, E=np.sqrt(2)):
    return np.nan_to_num(
        np.abs(u) * np.sqrt(u ** 2 - E ** 2) / E ** 2 - np.log(np.abs(u) + np.sqrt(u ** 2 - E ** 2)) + np.log(E),
        nan=np.inf)
