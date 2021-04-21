#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to cleaning methods ready for ML applications."""

from typing import Optional

import numpy as np
import pandas as pd

from turbopanda.str import pattern
from turbopanda.utils import belongs, instance_check, union, intersect
from turbopanda.pipe import zscore, clean1

__all__ = ("select_xcols", "preprocess_continuous_X", "preprocess_continuous_X_y")


def select_xcols(df: pd.DataFrame, xs, y):
    """Selects the appropriate x-column selection from a dataset for ML use."""
    if xs is None:
        return df.columns.difference(pd.Index([y]))
    elif isinstance(xs, str):
        return pattern(xs, df.columns)
    else:
        return xs


def preprocess_continuous_X(df, cols=None):
    """Preprocess dataframe into a machine-learning ready matrix X.

    Performs z-score transformation, cleaning of columns, removal of categorical types and missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset
    cols : list of str, optional
        Subset of the columns to choose. If None, uses all of the columns in df.

    Returns
    -------
    _df : pd.DataFrame
        Preprocessed dataset ready for ml uses.

    """
    if cols is None:
        cols = df.columns

    return (
        df[cols]
        .pipe(zscore)
        .pipe(clean1)
        .select_dtypes(exclude=["category", "object"])
        .dropna()
    )


def preprocess_continuous_X_y(df, xcols, ycols, for_sklearn=True):
    """Preprocess and split dataframe into X and y machine-learning ready datasets.

    Preprocesses especially for sklearn estimator object fit methods.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset
    xcols : list of str
        Subset of the columns to choose.
    ycols : str, list of str
        Subset of the columns for target.
    for_sklearn : bool, default=True
        Returns a np.ndarray if true, else pd.Series/DataFrame

    Returns
    -------
    _x : np.ndarray/pd.DataFrame
        Design matrix. X is reshaped ready for scikit-learn
    _y : np.ndarray/pd.Series
        Target variable
    """
    __data = preprocess_continuous_X(df, union(xcols, ycols))
    if for_sklearn:
        # returns np.ndarray objects properly configured
        _x = np.asarray(__data[xcols])
        _y = np.asarray(__data[ycols])
        if isinstance(xcols, str) or (
            isinstance(xcols, (list, tuple)) and len(xcols) == 1
        ):
            _x = _x.reshape(-1, 1)
        return _x, _y
    else:
        return __data[xcols], __data[ycols]
