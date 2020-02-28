#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs statistical function in conjunction with machine learning analysis."""

import numpy as np
import pandas as pd

from turbopanda.ml._clean import cleaned_subset
from turbopanda.utils import standardize, instance_check

__all__ = ('vif', 'cook_distance', 'hat', 'hat_generalized')


def hat(df, x):
    """Calculates the hat matrix.

    H = X(X^T X)^{-1} X^T

     Parameters
    ----------
    df : MetaPanda
        The full dataset
    x : list/tuple
        List of selected x columns


    Returns
    -------
    H : np.ndarray
        hat matrix.
    """

    X = np.atleast_2d(df[x].values)
    # standardize
    X = standardize(X)
    # calculate H
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    return H


def hat_generalized(df, x, cov):
    """Calculates the generalized hat matrix.

    H = X(X^T C^-1 X)^{-1} X^T C^-1

    Parameters
    ----------
    df : MetaPanda
        The full dataset
    x : list/tuple
        List of selected x columns
    cov : np.ndarray
        Conditioned covariance matrix

    Returns
    -------
    H : np.ndarray
        hat matrix.
    """
    instance_check(cov, np.ndarray)

    X = np.atleast_2d(df[x].values)
    # assume covariance matrix of errors
    X = standardize(X)
    # invert covariance matrix.
    c_inv = np.linalg.pinv(cov)
    H = X @ np.linalg.pinv(X.T @ c_inv @ X) @ X.T @ c_inv
    return H


def vif(df, x, y):
    """Calculates the variance inflationary factor for every X feature in df.

    Parameters
    ----------
    df : MetaPanda (n, p)
        The dataset
    x : selector
        A selection of x columns
    y : str
        The y column

    Returns
    -------
    vif : pd.Series (|x|,)
        variance inflationary factors for each in x
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

    instance_check(y, str)

    _df = cleaned_subset(df, x, y)
    _X = np.atleast_2d(_df[df.view(x)].values)
    # standardize
    _X = standardize(_X)

    if _X.shape[1] > 1:
        # for every column, extract vif
        vifs = [vif(_X, i) for i in range(_X.shape[1])]
        return pd.Series(vifs, index=df.view(x))
    else:
        return []


def cook_distance(df, x, y, yp):
    """Calculates cook's distance as a measure of outlier influence.

    Parameters
    ----------
    df : MetaPanda (n, p)
        The dataset
    x : selector
        A selection of x columns
    y : str
        The y column
    yp : np.ndarray (n, )
        The fitted values of y

    Returns
    --------
    c : pd.Series (n, )
        Cook's value for each in y.
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
