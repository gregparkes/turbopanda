#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs statistical function in conjunction with machine learning analysis."""

import numpy as np
import pandas as pd

from ._fit import cleaned_subset

__all__ = ('vif', 'cook_distance', 'hat', 'hat_generalized')


def hat(df, x):
    """Calculates the hat matrix.

    H = X(X^T X)^{-1} X^T

    Parameters
    ----------
    df : MetaPanda
    x : list/tuple

    Returns
    -------
    H : np.ndarray
        hat matrix.
    """
    X = np.atleast_2d(df[x].values)
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    return H


def hat_generalized(df, x, cov):
    """Calculates the generalized hat matrix.

    H = X(X^T C^-1 X)^{-1} X^T C^-1

    Parameters
    ----------
    df : MetaPanda
    x : list/tuple

    Returns
    -------
    H : np.ndarray
        hat matrix.
    """
    X = np.atleast_2d(df[x].values)
    # assume covariance matrix of errors
    c_inv = np.linalg.pinv(cov)
    H = X @ np.linalg.pinv(X.T @ c_inv @ X) @ X.T @ c_inv
    return H


def vif(df, x, y):
    """Calculates the variance inflationary factor for every X feature in df.

    Where df is MetaPanda, x is selector and y is str
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

    _df = cleaned_subset(df, x, y)
    _X = np.atleast_2d(_df[df.view(x)].values)
    if _X.shape[1] > 1:
        # for every column, extract vif
        vifs = [vif(_X, i) for i in range(_X.shape[1])]
        return pd.Series(vifs, index=df.view(x))
    else:
        return []


def cook_distance(df, x, y, yp):
    """Calculates cook's distance as a measure of outlier influence.

    Where df is Metapanda, x is selector and y is str.
    """
    # we clean df by choosing consistent subset, no NA.
    _df = cleaned_subset(df, x, y)
    _p = len(df.view(x))
    resid_sq = np.square(_df[y] - yp)

    # calculate hat matrix as OLS : @ is dot product between matrices
    diag_H = np.diag(hat(_df, df.view(x)))
    # calculate cook points
    cooks = (resid_sq / (_p * np.mean(resid_sq))) * (diag_H / np.square(1 - diag_H))
    return cooks
