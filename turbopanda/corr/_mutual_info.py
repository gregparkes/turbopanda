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



__all__ = ['mutual_info', 'entropy', 'conditional_entropy', 'continuous_mutual_info']


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

    Uses natural log for density normalization, so Shannon is in 'Nat' units.

        H(X) if just X,
        else H(X, Y) if Y is also given,
        else H(X, Y, Z) if Y and Z are given.

    By default, assumes density of X must be estimated, which we use np.histogram, given appropriate resolution to problem.

    Parameters
    ----------
    X : array_like, vector
        Continuous random variable X.
    Y : array_like, vector, optional
        Continuous random variable Y.
        If not None, computes joint entropy H(X, Y).
    Z : array_like, vector, optional
        Continuous random variable Z.
        If not None, computes joint entropy H(X, Y, Z). Y must also be present.

    Returns
    -------
    H : float
        Shannon entropy H(X) or H(X, Y) if Y is given, or H(X, Y, Z) if Y and Z are given.
    """
    # divide X into bins first
    # bin range
    instance_check(X, np.ndarray)
    instance_check(binned, bool)
    instance_check(normed, bool)

    if Z is not None:
        # prevent memory implosion.
        D = _estimate_density(X, Y, Z, r=1000)
    else:
        D = _estimate_density(X, Y)

    H = -np.sum(D * np.log2(D))
    return H


def conditional_entropy(X, Y):
    """Calculates conditional entropy H(X|Y) using Shannon entropy.

    H(X|Y) = H(X, Y) - H(Y)

    H(X,Y) = 0 if and only if the value of X is completely determined by the value of Y.

    In addition, H(X,Y) = H(X) if and only if X and Y are independent random variables.]

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
    """Given random continuous variables X, Y, determines the mutual information within.

    Entropy is estimated using Shannon entropy method. Provides for conditional calculations.

    I(X; Y) = H(X) + H(Y) - H(X, Y)  where Z is not present.
    I(X; Y|Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)  where Z is present

    Parameters
    ----------
    X : array_like, vector
        Continuous random variable X.
    Y : array_like, vector
        Continuous random variable Y.
    Z : array_like, vector, optional
        Continuous random variable Z to condition on.

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


@deprecated('0.2.2', "0.2.4", instead='discrete_mutual_info',
            reason='Does not work for continuous inputs, alternative in development.')
def mutual_info(data, x, y, z=None, **kws):
    """Calculates the mutual information between X, Y, and potentially Z.

    Parameters
    ---------
    data : pd.DataFrame / MetaPanda
        The full dataset.
    x : (str, list, tuple, pd.Index), optional
        Subset of input(s) for column names.
            if None, uses the full dataset. Y must be None in this case also.
    y : (str, list, tuple, pd.Index)
        Subset of output(s) for column names.
            if None, uses the full dataset (from optional `x` subset)
    z : (str, list, tuple, pd.Index), optional
        set of covariate(s). Covariates are needed to compute conditional mutual information.
            If None, uses standard MI.
    kws : dict
        Additional keywords to pass to drv.information_mutual[_conditional]

    Returns
    -------
    MI : np.ndarray
        mutual information matrix, (same dimensions as |x|, |y|, |z| input).
    """
    from pyitlib import discrete_random_variable as drv

    instance_check(data, MetaPanda)
    instance_check(x, (str, list, tuple, pd.Index))
    instance_check(y, (str, list, tuple, pd.Index))
    instance_check(z, (type(None), str, list, tuple, pd.Index))

    # downcast if list/tuple/pd.index is of length 1
    x = x[0] if (isinstance(x, (tuple, list, pd.Index)) and len(x) == 1) else x
    y = y[0] if (isinstance(y, (tuple, list, pd.Index)) and len(y) == 1) else y

    # cleaned set with no object or id columns
    _clean = data.view_not("object", '_id$', '_ID$', "^counter")
    # assume x, y, z are selectors, perform intersection between cleaned set and the columns we want
    _X = intersect(data.view(x), _clean)
    _Y = intersect(data.view(y), _clean)
    if z is not None:
        _Z = intersect(data.view(z), _clean)
        # calculate conditional mutual information
        _mi = drv.information_mutual_conditional(
            data[_X].T, data[_Y].T, data[_Z].T, cartesian_product=True, **kws
        )
    else:
        _mi = drv.information_mutual(
            data[_X].T, data[_Y].T, cartesian_product=True, **kws
        )

    # depending on the dimensions of x, y and z, can be up to 3D
    return _mi
