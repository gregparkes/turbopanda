#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defines functions to calculate the underlying density of 1:k-d float distributions."""

import numpy as np
from typing import Optional

from turbopanda.utils import instance_check, remove_na, nonnegative
from ._kde import freedman_diaconis_bins


def density(X: np.ndarray,
            Y: Optional[np.ndarray] = None,
            Z: Optional[np.ndarray] = None,
            r: Optional[int] = None) -> np.ndarray:
    """Estimates the density of X using binning, accepts np.ndarray.

    Parameters
    ----------
    X : np.ndarray (n,)
        The first dimension
    Y : np.ndarray (n,), optional
        The second dimension
    Z : np.ndarray (n,), optional
        The third dimension
    r : int, optional
        The number of bins for each dimension,
        If None, uses the freedman-diaconis rule

    Returns
    -------
    d : np.ndarray (r,...)
        The density in binned-dimensions
    """
    instance_check(X, np.ndarray)
    instance_check((Y, Z), (type(None), np.ndarray))
    instance_check(r, (type(None), int))

    if r is None:
        r = freedman_diaconis_bins(X)

    if Y is None and Z is None:
        _X = remove_na(X)
        return np.histogram(_X, bins=r, density=True)[0]
    elif Z is None:
        _X, _Y = remove_na(X, Y, paired=True)
        return np.histogram2d(_X, _Y, bins=(r, r), density=True)[0]
    else:
        return np.histogramdd(np.vstack((X, Y, Z)).T, bins=(r, r, r), density=True)[0]
