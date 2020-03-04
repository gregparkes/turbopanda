#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles some mutual information calculations."""

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from turbopanda._metapanda import MetaPanda
from turbopanda.utils import intersect, instance_check
from turbopanda._deprecator import deprecated

__all__ = ('entropy', 'conditional_entropy', 'continuous_mutual_info')


def _estimate_density(X, Y=None, Z=None, r=10000):
    """Estimates the density of X using binning."""
    if Y is None and Z is None:
        return np.histogram(X, bins=r, density=True)[0]
    elif Z is None:
        return np.histogram2d(X, Y, bins=(r, r), density=True)[0]
    else:
        return np.histogramdd(np.vstack((X, Y, Z)).T, bins=(r, r, r), density=True)[0]


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

    if Z is not None:
        # prevent memory implosion.
        D = _estimate_density(X, Y, Z, r=1000)
    else:
        D = _estimate_density(X, Y)

    H = -np.sum(D * np.log2(D))
    return H


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
    H_Y = entropy(Y)
    H_XY = entropy(X, Y)
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
    if Z is None:
        H_X = entropy(X)
        H_Y = entropy(Y)
        H_XY = entropy(X, Y)
        return H_X + H_Y - H_XY
    else:
        H_XZ = entropy(X, Z)
        H_YZ = entropy(Y, Z)
        H_XYZ = entropy(X, Y, Z)
        H_Z = entropy(Z)
        return H_XZ + H_YZ - H_XYZ - H_Z
