#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defines functions to calculate the underlying density of 1:k-d float distributions."""

import numpy as np
from typing import Optional
from numba import njit

from turbopanda.utils import instance_check, remove_na, nonnegative
from ._kde import freedman_diaconis_bins


@njit
def _density1d(X, r=10):
    n = X.shape[0]
    hist, bin_edges = np.histogram(X, bins=r)
    bin_diff = r / (bin_edges[r] - bin_edges[0])
    return hist / (n / bin_diff)


@njit
def _density2d(X, Y, r=10):
    xb = np.linspace(np.nanmin(X), np.nanmax(X), r+1)
    yb = np.linspace(np.nanmin(Y), np.nanmax(Y), r+1)
    hist = np.empty((r, r))
    normalizer = (X.shape[0] / (r / (xb[r] - xb[0]) * r / (yb[r] - yb[0])))

    for x in range(r):
        for y in range(r):
            Xn = np.bitwise_and(X >= xb[x], X <= xb[x+1])
            Yn = np.bitwise_and(Y >= yb[y], Y <= yb[y+1])
            hist[x, y] = np.nansum(np.bitwise_and(Xn, Yn))

    return hist / normalizer


def density(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    Z: Optional[np.ndarray] = None,
    r: Optional[int] = None,
) -> np.ndarray:
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
        r = min(freedman_diaconis_bins(X), 50)
    else:
        nonnegative(r, int)

    if Y is None and Z is None:
        _X = remove_na(X)
        return np.histogram(_X, bins=r, density=True)[0]
    elif Z is None:
        _X, _Y = remove_na(X, Y, paired=True)
        return np.histogram2d(_X, _Y, bins=(r,r), density=True)[0]
    else:
        return np.histogramdd(np.vstack((X, Y, Z)).T, bins=(r, r, r), density=True)[0]
