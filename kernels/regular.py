"""
Collection of rare kernel functions
"""

from kernels.base import Kernel
import numpy as np
from kernels.utils import euclidean_dist_matrix
import warnings

class Cossim(Kernel):
    """
    Cosine similarity kernel, 

        K(x, y) = <x, y> / (||x|| ||y||)

    """

    def _compute(self, data_1, data_2):
        self._dim = data_1.shape[1]
        norm_1 = np.sqrt((data_1 ** 2).sum(axis=1)).reshape(data_1.shape[0], 1)
        norm_2 = np.sqrt((data_2 ** 2).sum(axis=1)).reshape(data_2.shape[0], 1)
        return data_1.dot(data_2.T) / (norm_1 * norm_2.T)

    def dim(self):
        return self._dim

class Exponential(Kernel):
    """
    Exponential kernel, 

        K(x, y) = e^(-||x - y||/(2*s^2))

    where:
        s = sigma
    """

    def __init__(self, sigma=None):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = 2 * sigma**2

    def _compute(self, data_1, data_2):
        if self._sigma is None:
            # modification of libSVM heuristics
            self._sigma = float(data_1.shape[1])

        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return np.exp(-np.sqrt(dists_sq) / self._sigma)

    def dim(self):
        return np.inf


class Laplacian(Exponential):
    """
    Laplacian kernel, 

        K(x, y) = e^(-||x - y||/s)

    where:
        s = sigma
    """

    def __init__(self, sigma=None):
        self._sigma = sigma



class RationalQuadratic(Kernel):
    """
    Rational quadratic kernel, 

        K(x, y) = 1 - ||x-y||^2/(||x-y||^2+c)

    where:
        c > 0
    """

    def __init__(self, c=1):
        self._c = c

    def _compute(self, data_1, data_2):
        
        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return 1. - (dists_sq / (dists_sq + self._c))

    def dim(self):
        return None #unknown?


class InverseMultiquadratic(Kernel):
    """
    Inverse multiquadratic kernel, 

        K(x, y) = 1 / sqrt(||x-y||^2 + c^2)

    where:
        c > 0
    """

    def __init__(self, c=1):
        self._c = c ** 2

    def _compute(self, data_1, data_2):
        
        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return 1. / np.sqrt(dists_sq + self._c)

    def dim(self):
        return np.inf


class Cauchy(Kernel):
    """
    Cauchy kernel, 

        K(x, y) = 1 / (1 + ||x - y||^2 / s ^ 2)

    where:
        s = sigma
    """

    def __init__(self, sigma=None):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = sigma**2

    def _compute(self, data_1, data_2):
        if self._sigma is None:
            self._sigma = float(data_1.shape[1])

        dists_sq = euclidean_dist_matrix(data_1, data_2)

        return 1 / (1 + dists_sq / self._sigma)

    def dim(self):
        return np.inf



class TStudent(Kernel):
    """
    T-Student kernel, 

        K(x, y) = 1 / (1 + ||x - y||^d)

    where:
        d = degree

    """

    def __init__(self, degree=2):
        self._d = degree

    def _compute(self, data_1, data_2):

        dists = np.sqrt(euclidean_dist_matrix(data_1, data_2))
        return 1 / (1 + dists ** self._d)

    def dim(self):
        return None


class ANOVA(Kernel):
    """
    ANOVA kernel, 
        K(x, y) = SUM_k exp( -sigma * (x_k - y_k)^2 )^d
    """

    def __init__(self, sigma=1., d=2):
        self._sigma = sigma
        self._d = d

    def _compute(self, data_1, data_2):

        kernel = np.zeros((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            kernel += np.exp( -self._sigma * (column_1 - column_2.T)**2 ) ** self._d

        return kernel

    def dim(self):
        return None


def default_wavelet(x):
    return np.cos(1.75*x)*np.exp(-x**2/2)

class Wavelet(Kernel):
    """
    Wavelet kernel,

        K(x, y) = PROD_i h( (x_i-c)/a ) h( (y_i-c)/a )

    or for c = None

        K(x, y) = PROD_i h( (x_i - y_i)/a )

    """

    def __init__(self, h=default_wavelet, c=None, a=1):
        self._c = c
        self._a = a
        self._h = h

    def _compute(self, data_1, data_2):

        kernel = np.ones((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            if self._c is None:
                kernel *= self._h( (column_1 - column_2.T) / self._a )
            else:
                kernel *= self._h( (column_1 - self._c) / self._a ) * self._h( (column_2.T - self._c) / self._a )

        return kernel

    def dim(self):
        return None


class Fourier(Kernel):
    """
    Fourier kernel,

        K(x, y) = PROD_i (1-q^2)/(2(1-2q cos(x_i-y_i)+q^2))
    """

    def __init__(self, q=0.1):
        self._q = q

    def _compute(self, data_1, data_2):

        kernel = np.ones((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            kernel *= (1-self._q ** 2) / \
                      (2.*(1. - 2.*self._q *np.cos(column_1 - column_2.T) + self._q ** 2))

        return kernel

    def dim(self):
        return None

class Tanimoto(Kernel):
    """
    Tanimoto kernel
        K(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)

    """
    def _compute(self, data_1, data_2):

        norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
        norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
        prod = data_1.dot(data_2.T)
        return prod / (norm_1 + norm_2.T - prod)

    def dim(self):
        return None


class Sorensen(Kernel):
    """
    Sorensen kernel
        K(x, y) = 2 <x, y> / (||x||^2 + ||y||^2)

    """
    def _compute(self, data_1, data_2):

        norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
        norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
        prod = data_1.dot(data_2.T)
        return 2 * prod / (norm_1 + norm_2.T)

    def dim(self):
        return None




class GeneralizedHistogramIntersection(Kernel):
    """
    Generalized histogram intersection kernel
        K(x, y) = SUM_i min(|x_i|^alpha, |y_i|^alpha)

    """

    def __init__(self, alpha=1.):
        self._alpha = alpha

    def _compute(self, data_1, data_2):

        return Min()._compute(np.abs(data_1)**self._alpha,
                              np.abs(data_2)**self._alpha)

    def dim(self):
        return None






