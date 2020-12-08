from functools import partial
import numpy as np

from .constants import integration_region


def in_integration_domain(x, x1, P=1, Q=1, R=1):
    """
    Check if (x, x1) is in the integration region for the complexity.
    :param x: float; inegration var x
    :param x1: float; integration var x1
    :param P: float;
    :param Q: float;
    :param R: float;
    :return: bool
    """
    boundary1 = x < P
    boundary2 = x1 > (R * x + Q)
    return boundary1 & boundary2


def theta(x, x1, exponent_vals, p, q):
    """
    Compute theta, the expected log complexity leading term, from precomputed values of the exponent function in two
    vars (x, x1).
    :param x: np.ndarray of shape (n_points, n_points); x values on a grid where exponent has been computed
    :param x1: np.ndarray of shape (n_points, n_points); x1 values on a grid where exponent has been computed
    :param exponent_vals: np.ndarray of shape (n_points, n_points); computed exponent function values
    :param p: int; number of discriminator layers
    :param q: int; number of generator layers
    :return: np.ndarray of shape (n_points, n_points); grid of uD values for evaluating theta
             np.ndarray of shape (n_points, n_points); grid of uG values for evaluating theta
             np.ndarray of shape (n_points, n_points); grid  evaluated theta values
    """
    R = -p / ((p + q) * 2 ** (p + q))
    uG_max = x.max() / (2 ** (p + q) * (p + q))
    uG_min = x.min() / (2 ** (p + q) * (p + q))

    uD_max = (x.max() * R - x1.min()) / p
    uD_min = (x.min() * R - x1.max()) / p

    uD = np.linspace(uD_min, uD_max, 500)
    uG = np.linspace(uG_min, uG_max, 500)
    uD, uG = np.meshgrid(uD, uG)

    thetas = np.zeros_like(uD)

    for col_ind in range(uD.shape[1]):
        for row_ind in range(uG.shape[0]):
            P, Q, R = integration_region(uD[row_ind, col_ind], uG[row_ind, col_ind], p, q)
            in_domain = partial(in_integration_domain, P=P, Q=Q, R=R)
            in_vals = exponent_vals[in_domain(x, x1)]
            thetas[row_ind, col_ind] = np.max(in_vals) if len(in_vals) else np.nan
    return uD, uG, thetas
