#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A fast way to calculate Linear or Logistic Regressions for continuous/discrete variables."""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from typing import Union
from sklearn.linear_model import LogisticRegression

from turbopanda.utils import is_dataframe_float, is_column_discrete


def _ordinary_least_squares(X, y):
    """Where X is continuous, y is continuous. X is of shape {n, p}, y is of shape {n,},
    Returns predictions, residuals"""
    _X = zscore(np.asarray(X))
    _y = np.asarray(y)
    if _X.ndim == 1:
        _X = np.atleast_2d(_X).T
    _beta = np.linalg.lstsq(_X, _y, rcond=None)[0]
    yp = np.dot(_X, _beta)
    return yp, _y-yp


def _ordinary_logistic_reg(X, y):
    """Where X is continuous, y is binary. X is of shape {n, p}, y is of shape {n,},
    Returns predictions, residuals."""
    _X = zscore(np.asarray(X))
    _y = np.asarray(y)
    # standardize X if X is continuous...
    if _X.ndim == 1:
        _X = np.atleast_2d(_X).T
    # build model
    lr = LogisticRegression().fit(_X, _y)
    yp = lr.predict(_X)
    # return weights
    return yp, np.not_equal(_y, yp).astype(np.uint8)


def lm(X: Union[np.ndarray, pd.Series, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
    """Creates a basic linear model between X and y depending on the state it is in.

    Parameters
    ----------
    X : ndarray, Series, DataFrame
        The input column or matrix.
    y : ndarray, Series
        The target column

    Returns
    -------
    yp : ndarray, Series
        The predicted values
    """
    # check that every column in X is continuous
    _X = np.asarray(X)
    if not is_dataframe_float(_X):
        raise TypeError("`X` variables in `stats.lm` all must be of type `float`/continuous.")
    _y = np.asarray(y)

    # check if y is discrete
    if is_column_discrete(_y):
        return _ordinary_logistic_reg(_X, _y)
    else:
        return _ordinary_least_squares(_X, _y)
