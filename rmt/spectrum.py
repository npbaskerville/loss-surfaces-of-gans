import numpy as np


def density(x, x1=0, kappa=0.5, b=1, b1=1):
    """
    Compute the limiting spectral density of the random matrix H at the point x.

    :param x1: float; the constant shift in the upper left block of the matrix
    :param kappa: float in (0,1); ratio of the dimension of the upper left block to the whole matrix
    :param b: float; coefficient of the full rank GOE
    :param b1: float; coefficient of the upper left GOE

    :return: float
    """
    norm = 2 / (np.pi * b ** 2)
    k = kappa
    kp = (1 - kappa)

    # coefficients of the t quadratics (A, B, C), (D, E, F)
    coeff_A = 1
    coeff_B = -x
    coeff_C = kp * b ** 2 / 2
    coeff_D = 1 + b1 ** 2 / b ** 2 / k
    coeff_E = -(b1 ** 2 / b ** 2 / k * x - x1)
    coeff_F = -kp / k * b1 ** 2 / 2
    coeffs = [coeff_A * coeff_D,
              coeff_A * coeff_E + coeff_B * coeff_D,
              coeff_A * coeff_F + coeff_C * coeff_D + coeff_B * coeff_E + k * b ** 2 / 2,
              coeff_B * coeff_F + coeff_C * coeff_E,
              coeff_C * coeff_F]

    roots = np.roots(coeffs)
    return np.max(np.maximum(np.array([-np.imag(r) for r in roots]) * norm, 0))


def support(x1, b, b1):
    """
    We do not theoretically know the support of the LSD in closed form, but this function constructs a good guess,
    which certainty always contains the LSD support and is adequate for plotting and quadrature.
    :param x1: float; shift of the upper left block
    :param b: float; coeff of the full rank GOE
    :param b1: float; coeff of the partial rank GOE
    :return: tuple[float]; lower, upper limits of domain
    """
    lims = np.array([[-2 * b - x1, 2 * b - x1], [-2 * b1, 2 * b1]])
    lims = (np.min(lims[:, 0]), np.max(lims[:, 1]))
    return lims
