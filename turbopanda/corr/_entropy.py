#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles some mutual information calculations."""

# future imports
from __future__ import absolute_import, division, print_function

import numpy as np
from typing import Optional

from turbopanda.stats import density
from turbopanda.stats._kde import freedman_diaconis_bins
from turbopanda.utils import instance_check, as_flattened_numpy

__all__ = ("entropy", "conditional_entropy", "continuous_mutual_info")


def _entropy1d(X, bins, eps=1e-12):
    D = np.histogram(X, bins=bins, density=True)[0]
    return -np.sum(D * np.log2(D + eps))


def _entropy2d(X, Y, bins, eps=1e-12):
    D = np.histogram2d(X, Y, bins=bins, density=True)[0]
    return -np.sum(D * np.log2(D + eps))


def _fast_entropy(X, Y=None, Z=None, r=None):
    if r is None:
        r = min(freedman_diaconis_bins(X), 50)

    if Y is None and Z is None:
        return _entropy1d(X, bins=r)
    elif Z is None:
        return _entropy2d(X, Y, bins=r)
    else:
        D = density(X, Y, Z, r)
        return -np.sum(D * np.log2(D + eps))


def entropy(
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        r: Optional[int] = None
) -> float:
    """Calculates shannon entropy given X.

    Uses log2 for density normalization.`X`,`Y` and `Z` must all be continuous.

    By default, assumes density of `X` must be estimated,
        which we use np.histogram, given appropriate resolution to problem.

    Parameters
    ----------
    X : array_like, vector
        Continuous random variable X.
    Y : array_like, vector, optional
        Continuous random variable Y.
        If not None, computes joint entropy :math:`H(X, Y)`.
    Z : array_like, vector, optional
        Continuous random variable Z.
        If not None, computes joint entropy :math:`H(X, Y, Z)`. Y must also be present.
    r : int, optional
        Determines the bin size for density calculations

    Returns
    -------
    H : float
        Shannon entropy :math:`H(X)` or :math:`H(X, Y)` if `Y` is given, or :math:`H(X, Y, Z)` if `Y` and `Z` are given.
    """
    # divide X into bins first
    # bin range
    instance_check(X, np.ndarray)
    instance_check(Y, (type(None), np.ndarray))
    instance_check(Z, (type(None), np.ndarray))

    return _fast_entropy(as_flattened_numpy(X), Y, Z, r)


def conditional_entropy(X: np.ndarray, Y: np.ndarray) -> float:
    """Calculates conditional entropy H(X|Y) using Shannon entropy.

    .. math:: H(X|Y) = H(X, Y) - H(Y)

    where :math:`H(X,Y) = 0` if and only if the value of `X` is completely
        determined by the value of `Y`.

    In addition, :math:`H(X,Y) = H(X)` if and only if `X` and `Y` are
        independent random variables.

    Parameters
    ----------
    X : array_like
        The vector/matrix to calculate entropy on.
    Y : array_like
        The vector/matrix to calculate entropy on.

    Returns
    -------
    H : float
        H(X|Y) the entropy of X given Y
    """
    _X = as_flattened_numpy(X)
    _Y = as_flattened_numpy(Y)
    # cast entropy and return
    H_Y = _fast_entropy(_Y)
    H_XY = _fast_entropy(_X, _Y)
    return H_XY - H_Y


def continuous_mutual_info(
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None
) -> float:
    """Determines mutual information given random variables.

    .. math:: I(X;Y) = H(X) + H(Y) - H(X,Y)

    where :math:`H(X,Y)` is the joint probability, and :math:`H(X)` is the marginal distribution
        of X.

    Entropy is estimated using Shannon entropy method.
        Provides for conditional calculations.

    Parameters
    ----------
    X : array_like, vector
        Continuous random variable X.
    Y : array_like, vector
        Continuous random variable Y.
    Z : array_like, vector, optional
        Continuous random variable Z to condition on.

    Returns
    -------
    MI : float
        I(X; Y) or I(X; Y|Z)
    """
    _X = as_flattened_numpy(X)
    _Y = as_flattened_numpy(Y)

    r = min(freedman_diaconis_bins(_X), 50)

    if Z is None:
        # calculate
        H_X = _fast_entropy(_X, r=r)
        H_Y = _fast_entropy(_Y, r=r)
        H_XY = _fast_entropy(_X, _Y, r=r)
        return H_X + H_Y - H_XY
    else:
        _Z = as_flattened_numpy(Z)
        # calculate entropy.
        H_XZ = _fast_entropy(_X, _Z, r=r)
        H_YZ = _fast_entropy(_Y, _Z, r=r)
        H_XYZ = _fast_entropy(_X, _Y, _Z, r)
        H_Z = _fast_entropy(_Z, r=r)
        return H_XZ + H_YZ - H_XYZ - H_Z
