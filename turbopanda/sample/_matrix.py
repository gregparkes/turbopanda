#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for sampling data matrices with homoscedastic and heteroscedastic noise."""

import numpy as np
import warnings


def _matrix_creation(n_samples, n_features, rank, seed):
    if n_features > n_samples:
        raise ValueError("n_features must be <= n_samples")

    if rank is None:
        rank = n_features

    if rank > n_features:
        raise ValueError("rank must be <= n_features")

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # perform svd
    U, _, _ = np.linalg.svd(rng.randn(n_features, n_features))
    X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    return X, rng


def matrix_homo(n_samples: int,
                n_features: int,
                rank: int = None,
                sigma: float = 1.,
                seed: int = None):
    """Generates a data matrix X with homoscedastic noise.

    Data is drawn from Gaussian distributions according to sizes provided.

    See https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html

    Data is generated as:
        Sample T ~ N(0, 1): of size (n_features, n_features)
        Fetch orthonormal matrix U = svd(T)
        Sample R ~ N(0, 1): of size (n_samples, rank)
        Sample E ~ N(0, sigma)
        X = dot(R, U.T) + E: selecting up to [:rank] columns in U

    Parameters
    ----------
    n_samples : int
        The number of data points
    n_features : int
        The number of dimensions. Must be <= n_samples
    rank : int, optional
        The true underlying dimensionality of data. Must be <= n_features.
        If rank==None, rank = n_features
    sigma : float, optional
        Homoscedastic noise parameter. Must be > 0.
    seed : int, optional
        Random number seed for consistent matrices.

    Returns
    -------
    X : np.ndarray (n_samples, n_features)
        Data matrix.
    """
    # perform checks in matrix creation
    X, rng = _matrix_creation(n_samples, n_features, rank, seed)
    # make it homoscedastic
    X_homo = X + sigma * rng.randn(n_samples, n_features)
    return X_homo


def matrix_hetero(n_samples: int,
                  n_features: int,
                  rank: int = None,
                  sigma: float = 1.,
                  seed: int = None):
    """Generates a data matrix X with heteroscedastic noise.

    Data is drawn from Gaussian distributions according to sizes provided.

    See https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html

    Data is generated as:
        Sample T ~ N(0, 1): of size (n_features, n_features)
        Fetch orthonormal matrix U = svd(T)
        Sample R ~ N(0, 1): of size (n_samples, rank)
        Sample S ~ U(0, 1): of size (n_features,)
        Sample E ~ sigma x S + (sigma / 2)
        X = dot(R, U.T) + N(0, 1).E: selecting up to [:rank] columns in U

    Parameters
    ----------
    n_samples : int
        The number of data points
    n_features : int
        The number of dimensions. Must be <= n_samples
    rank : int, optional
        The true underlying dimensionality of data. Must be <= n_features.
        If rank==None, rank = n_features
    sigma : float or np.ndarray, optional
        Homoscedastic noise parameter. Must be > 0. If array, must be of length n_features
    seed : int, optional
        Random number seed for consistent matrices.

    Returns
    -------
    X : np.ndarray (n_samples, n_features)
        Data matrix.
    """
    # perform checks in matrix creation
    X, rng = _matrix_creation(n_samples, n_features, rank, seed)
    # check sigma and convert to appropriate type
    if isinstance(sigma, np.ndarray) and sigma.shape[0] == n_features:
        sigmas = sigma
    else:
        sigmas = sigma * rng.rand(n_features) + sigma / 2.
    # make it heteroscedastic
    X_hetero = X + rng.randn(n_samples, n_features) * sigmas
    return X_hetero
