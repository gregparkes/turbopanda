#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs statistical function in conjunction with machine learning analysis."""

import numpy as np
import pandas as pd

from turbopanda._dependency import is_statsmodels_installed
from turbopanda.ml._clean import preprocess_continuous_X_y, select_xcols
from turbopanda.utils import instance_check

__all__ = ('vif', 'cook_distance', 'hat', 'leverage')


def _hat_ols(X):
    """Uses the ordinary-least squares version of hat matrix."""
    X = np.atleast_2d(X)
    # attempt to invert X.T @ X
    X_inv = np.linalg.pinv(X.T @ X)
    # calculate H
    H = X @ X_inv @ X.T
    return H


def _hat_ridge(X, alpha=1.):
    """Uses the ridge direct version of hat matrix."""
    X = np.atleast_2d(X)
    # attempt to invert X.T @ X
    X_inv = np.linalg.pinv((X.T @ X) + (alpha * np.eye(X.shape[1])))
    H = X @ X_inv @ X.T
    return H


def hat(X, method="ols"):
    """Calculates the hat or projection `H` matrix.

    Ordinary least squares version:
        .. math:: H = X(X^T X)^{-1} X^T

    Parameters
    ----------
    X : np.ndarray
        numpy matrix.
    method : str
        Choose from {'ols', 'ridge'}

    Returns
    -------
    H : np.ndarray
        hat matrix.
    """
    if method == "ols":
        return _hat_ols(X)
    elif method == 'ridge':
        return _hat_ridge(X)
    else:
        raise ValueError("method '{}' not found in {'ols', 'ridge'}")


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

    if is_statsmodels_installed():
        from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif

        instance_check(y, str)
        _df = df.df_ if not isinstance(df, pd.DataFrame) else df
        _xcols = select_xcols(df, x, y)
        _x, _y = preprocess_continuous_X_y(_df, _xcols, y)

        if _x.shape[1] > 1:
            # for every column, extract vif
            vifs = [_vif(_x, i) for i in range(_x.shape[1])]
            return pd.Series(vifs, index=_xcol)
        else:
            return []
    else:
        return []


def leverage(X):
    """Calculates the leverage points from the projection matrix."""
    return np.diag(hat(X))


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
    _df = df.df_ if not isinstance(df, pd.DataFrame) else df
    _xcols = select_xcols(df, x, y)
    _x, _y = preprocess_continuous_X_y(_df, _xcols, y)

    _p = len(_xcol)
    # determine squared-residuals
    resid_sq = np.square(_y - yp)
    # calculate hat matrix as OLS : @ is dot product between matrices
    diag_H = leverage(_x)
    # calculate cook points
    cooks = (resid_sq / (_p * np.mean(resid_sq))) * (diag_H / np.square(1 - diag_H))
    return cooks
