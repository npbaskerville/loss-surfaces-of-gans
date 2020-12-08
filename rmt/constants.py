import numpy as np


def b_constants(p, q, sigma_z, kappa):
    """

    :param p: int; the number of layers in the discriminator (number of spins in its spin glass)
    :param q: int; the number of layers in the generator (number of spins in its spin glass)
    :param sigma_z: float; the variance of the generator latent noise
    :param kappa: float; ratio of the upper left block dimension to the whole matrix dimension
    :return: float; b -- coeff of full rank GOE
             float; b1 -- coeff of partial rank GOE (upper left block)
    """
    b1 = np.sqrt(p * (p - 1) * kappa)
    b = np.sqrt((p + q) * (p + q - 1) * 2 ** (p + q)) * sigma_z
    return b, b1


def complexity_constant(p, q, sigma_z, kappa):
    """
    Log of the constant term in front of the complexity.

    :param p: int; the number of layers in the discriminator (number of spins in its spin glass)
    :param q: int; the number of layers in the generator (number of spins in its spin glass)
    :param sigma_z: float; the variance of the generator latent noise
    :param kappa: float; ratio of the upper left block dimension to the whole matrix dimension
    :return: float
    """
    return np.log(2) * 13 / 2 + 5 * np.log(np.pi) - kappa / 2 * np.log(p + sigma_z ** 2 * (p + q) * 2 ** (p + q)) - (
            1 - kappa) / 2 * np.log(sigma_z ** 2 * (p + q) * 2 ** (p + q)) - kappa * np.log(kappa) / 2 - \
           (1 - kappa) * np.log(1 - kappa) / 2


def univariate_gaussian_variances(p, q, sigma_z):
    """
    Compute the variances of the x and x1 integration variables.

    :param p: int; the number of layers in the discriminator (number of spins in its spin glass)
    :param q: int; the number of layers in the generator (number of spins in its spin glass)
    :param sigma_z: float; the variance of the generator latent noise
    :return: float; variance of x
             float; variance of x_1
    """
    s_sq = sigma_z ** 2 * (p + q) ** 2 * 2 ** (3 * (p + q)) / 2
    s1_sq = p ** 2 / 2
    return s_sq, s1_sq


def integration_region(uD, uG, p, q):
    """
    Compute the constants in the definition of the integration region over (x, x1).
    :param uD: float; upper bound on discrimnator loss
    :param uG: float; upper bound on generator loss
    :param p: int; number of discriminator layers
    :param q: int; number of generator layers
    :return: float; P
             float; Q
             float: R
    """
    P = (p + q) * 2 ** (p + q) * uG
    Q = -p * uD
    R = -p / (2 ** (p + q) * (p + q))
    return P, Q, R
