#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for sampling correlated covariance matrices, and multivariate gaussian distributions."""

import numpy as np
import warnings

from turbopanda.utils import nonnegative, instance_check, arrays_equal_size
from turbopanda._dependency import is_numba_installed


def _create_cov_matrix(p, c, diag_v, rng_dir):
    # wrapper function to surround using jit.
    if is_numba_installed():
        # import
        from numba import jit

        @jit(nopython=True)
        def _generate_covariance_withjit():
            # diag elements
            _diag = np.full(p, diag_v) + np.random.normal(0, 0.005, p)
            _diag_mat = np.full((p, p), _diag)

            _cov_min = np.minimum(_diag_mat, _diag_mat.T) * c

            # a matrix to flip random directions.
            if rng_dir:
                # randomly -1 or 1.
                _dir = np.sign(np.random.uniform(-1, 1))
            else:
                _dir = np.sign(corr_ratio)

            if _dir < 0:
                # create -1s at offset positions to ensure invertibility
                _tmpX = np.ones((p, p))
                _tmpX[::2, ::2] = -1.
                _tmpX[1::2, 1::2] = -1.
                # alternative pos and negative diag terms
                _cov_lower = np.tril(_cov_min * _tmpX, k=-1)
            else:
                # all positive values.
                _cov_lower = np.tril(np.abs(_cov_min), k=-1)

            # make full cov matrix
            _cov_total = _cov_lower + _cov_lower.T + np.diag(_diag)
            return _cov_total
        return _generate_covariance_withjit()
    else:
        def _generate_covariance_woutjit():
            # diag elements
            _diag = np.full(p, diag_v) + np.random.normal(0, 0.005, p)
            _diag_mat = np.full((p, p), _diag)

            _cov_min = np.minimum(_diag_mat, _diag_mat.T) * c

            # a matrix to flip random directions.
            if rng_dir:
                # randomly -1 or 1.
                _dir = np.sign(np.random.uniform(-1, 1))
            else:
                _dir = np.sign(corr_ratio)

            if _dir < 0:
                # create -1s at offset positions to ensure invertibility
                _tmpX = np.ones((p, p))
                _tmpX[::2, ::2] = -1.
                _tmpX[1::2, 1::2] = -1.
                # alternative pos and negative diag terms
                _cov_lower = np.tril(_cov_min * _tmpX, k=-1)
            else:
                # all positive values.
                _cov_lower = np.tril(np.abs(_cov_min), k=-1)

            # make full cov matrix
            _cov_total = _cov_lower + _cov_lower.T + np.diag(_diag)
            return _cov_total
        return _generate_covariance_woutjit()


def covariance_matrix(p,
                      corr_ratio=0.5,
                      diag_var=1.,
                      random_direction=False):
    """Generates a randomly-generated 'correlated' covariance matrix.

    This is useful in situations where you want to create correlated synthetic
    data to test an algorithm.

    Bare in mind that `corr_ratio` follows the relationship rho = [1/p-1, 1], so negative
    correlations will be clipped at higher dimensions to ensure semipositive definite structures.

    Parameters
    ----------
    p : int
        The number of dimensions. p must be >= 2
    corr_ratio : float [-1..1]
        The proportion of 'correlation' within the matrix; 0 no correlation, 1 full positive correlation
            and -1 full negative correlation.
    diag_var : float
        The values on the diagonal, with small error (5e-03)
    random_direction : bool, default=False
        Correlation is randomly positive or negative if True, else uses the sign(corr_ratio).

    Returns
    -------
    cov : np.ndarray((p, p))
        Covariance matrix
    """
    nonnegative(p, int)
    # p must be greater than 1 to be multivariate gaussian.
    if p < 2:
        raise ValueError("'p' must be > 1")
    # clips ratio into the range [0, 1]
    _corr_ratio = np.clip(corr_ratio, 1. / (p - 1), 0.999)

    if not np.isclose(corr_ratio, _corr_ratio):
        warnings.warn("`corr_ratio` parameter is clipped from {:0.3f} to [{:0.3f}, 1]".format(
            corr_ratio, _corr_ratio
        ))

    return _create_cov_matrix(p, _corr_ratio, diag_var, random_direction)


def multivariate_gaussians(n, k, C=0.5):
    """Creates k multivariate Gaussian distributions with sample size n, according to correlations C.

    All Gaussians are with mu = 0, sigma = [ratio of C].

    Parameters
    ----------
    n : int
        Sample size
    k : int, list of int
        Dimensionality of one group (int) or each group of Multivariate Gaussians (list of int)
    C : float, list of float
        Correlation strength [-1...1] for all groups (float) or each group (list of float)

    Returns
    -------
    X : np.ndarray (n, sum(k))
        Multivariate Gaussian synthetic data
    """
    if n < 1:
        raise ValueError("'n' must be > 0")

    instance_check(k, (int, np.int, list, tuple))
    instance_check(C, (float, np.float, list, tuple))
    # if C is a list, ensure k and C are same length
    if isinstance(C, (list, tuple)) and isinstance(k, (list, tuple)):
        arrays_equal_size(k, C)

    # handle single k case
    if isinstance(k, (int, np.int)):
        # just one gaussian group
        if isinstance(C, (float, np.float)):
            return np.random.multivariate_normal(np.zeros(k),
                                                 covariance_matrix(k, C, random_direction=False),
                                                 size=n)
        else:
            raise ValueError("'C' must be of type 'float' when 'k' is of type 'int'")
    else:
        # must be a list, iterate over it
        result = []
        for i, p in enumerate(k):
            mu = np.zeros(p)
            # collect the correlation ratio if its a float or from a list
            c = C if isinstance(C, (float, np.float)) else C[i]
            # compute covariance matrix
            cov = covariance_matrix(p, c, random_direction=False)
            # make data
            X = np.random.multivariate_normal(mu, cov, size=n)
            result.append(X)

        return np.hstack(result)
