from joblib import Parallel, delayed
import numpy as np


def sample_spectrum(dimension=500, x1=1, kappa=0.3, b=1, b1=1):
    """
    Sample once from matrix distribution H with given dimension and parameters.

    :param dimension: int; dimension of matrices to sample
    :param x1: float; shift of upper left block of matrices
    :param kappa: float; ratio of upper left block to whole matrix dimension
    :param b: float; coefficient of full rank GOE
    :param b1: float; coefficient of partial rank GOE (upper left)
    :return: np.ndarray of shape (dimension, )[; eigenvalues of sampled matrix
    """
    dimension_discriminator = int(kappa * dimension)
    dimension_generator = dimension - dimension_discriminator
    matrix_1 = np.random.randn(dimension, dimension) / np.sqrt(dimension)
    matrix_1 = b * (matrix_1 + matrix_1.T) / 2
    matrix_2 = np.random.randn(dimension_discriminator, dimension_discriminator) / np.sqrt(dimension_discriminator)
    matrix_2 = b1 * (matrix_2 + matrix_2.T) / 2
    padded_matrix_2 = np.block(
        [[matrix_2 - x1 * np.eye(dimension_discriminator), np.zeros((dimension_discriminator, dimension_generator))],
         [np.zeros((dimension_generator, dimension_discriminator)),
          np.zeros((dimension_generator, dimension_generator))]])
    spectrum = np.linalg.eigvalsh(matrix_1 + padded_matrix_2)
    return spectrum


def sample_spectra(n_samples=100, **kwargs):
    """
    Sample multiple times from the matrix H distribution with a given dimension and parameters.
    :param n_samples: how many matrices to draw
    :param kwargs:
    :return: np.ndarray of shape (n_samples, dimension); eigenvalues of the sampled matrices
    """
    iterator = (delayed(sample_spectrum)(**kwargs) for _ in range(n_samples))
    return np.array(Parallel(n_jobs=-1, prefer='threads')(iterator))


def mc_log_determinant(x, x1, b=1, b1=1, kappa=0.5, n_samples=100, dimension=300):
    """
    Approximate the log expected determinant of the Hessian.

    :param x: float; shift of whole H matrix
    :param x1: float; shift of upper left block of matrices
    :param b: float; coefficient of full rank GOE
    :param b1: float; coefficient of partial rank GOE (upper left)
    :param kappa: float; ratio of upper left block to whole matrix dimension
    :param n_samples: how many matrices to draw
    :param dimension: int; dimension of matrices to sample
    :return: float;
    """
    spectra = sample_spectra(n_samples=n_samples, dimension=dimension, kappa=kappa, x1=x1, b=b, b1=b1)
    return np.log(np.mean(np.prod(np.abs(spectra - x), axis=1)))/dimension
