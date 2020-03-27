#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles some mutual information calculations."""

# future imports
from __future__ import absolute_import, division, print_function

import numpy as np

from turbopanda.utils import instance_check, remove_na

__all__ = ('entropy', 'conditional_entropy', 'continuous_mutual_info')


def _estimate_density(X, Y=None, Z=None, r=10000):
    """Estimates the density of X using binning, accepts np.ndarray. """
    if Y is None and Z is None:
        _X = remove_na(X)
        return np.histogram(_X, bins=r, density=True)[0]
    elif Z is None:
        _X, _Y = remove_na(X, Y, paired=True)
        return np.histogram2d(_X, _Y, bins=(r, r), density=True)[0]
    else:
        return np.histogramdd(np.vstack((X, Y, Z)).T, bins=(r, r, r), density=True)[0]


def _estimate_entropy(X, Y=None, Z=None):
    if Z is not None:
        # prevent memory implosion.
        D = _estimate_density(X, Y, Z, r=1000)
    else:
        D = _estimate_density(X, Y)
    return -np.sum(D * np.log2(D))


def entropy(X, Y=None, Z=None):
    """Calculates shannon entropy given X.

    Uses log2 for density normalization. `X`, `Y` and `Z` must all be continuous.

    .. math:: H(X)=-\sum_x p(x)\log_2 p(x)

    where :math:`H(X)` if just `X`, the joint entropy :math:`H(X, Y)` is calculated if Y is also given,
    and likewise :math:`H(X, Y, Z)` if `Y` and `Z` are given.

    By default, assumes density of `X` must be estimated, which we use np.histogram, given appropriate resolution to problem.

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

    return _estimate_entropy(X, Y, Z)


def conditional_entropy(X, Y):
    """Calculates conditional entropy H(X|Y) using :math:`\log_2` Shannon entropy.

    .. math:: H(X|Y) = H(X, Y) - H(Y)

    where :math:`H(X,Y) = 0` if and only if the value of `X` is completely determined by the value of `Y`.

    In addition, :math:`H(X,Y) = H(X)` if and only if `X` and `Y` are independent random variables.

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
    _X = np.asarray(X)
    _Y = np.asarray(Y)
    # cast entropy and return
    H_Y = _estimate_entropy(_Y)
    H_XY = _estimate_entropy(_X, _Y)
    return H_XY - H_Y


def continuous_mutual_info(X, Y, Z=None):
    """Given random continuous variables `X`, `Y`, determines the mutual information within.

    Entropy is estimated using Shannon entropy method. Provides for conditional calculations.

    .. math::
        I(X; Y) = H(X) + H(Y) - H(X, Y) \\
        I(X; Y|Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
    
    where `Z` may or may not be present, respectively.
    
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
    _X = np.asarray(X)
    _Y = np.asarray(Y)

    if Z is None:
        # calculate
        H_X = _estimate_entropy(_X)
        H_Y = _estimate_entropy(_Y)
        H_XY = _estimate_entropy(_X, _Y)
        return H_X + H_Y - H_XY
    else:
        _Z = np.asarray(Z)
        # calculate entropies.
        H_XZ = _estimate_entropy(_X, _Z)
        H_YZ = _estimate_entropy(_Y, _Z)
        H_XYZ = _estimate_entropy(_X, _Y, _Z)
        H_Z = _estimate_entropy(_Z)
        return H_XZ + H_YZ - H_XYZ - H_Z
