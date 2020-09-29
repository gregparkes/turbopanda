#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to cleaning methods ready for ML applications."""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, power_transform, quantile_transform, scale

from turbopanda._deprecator import deprecated
from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.str import pattern
from turbopanda.utils import belongs, instance_check, union, intersect
from turbopanda.pipe import zscore, clean1


__all__ = ('select_xcols', 'preprocess_continuous_X', 'preprocess_continuous_X_y')


def select_xcols(df: pd.DataFrame, xs, y):
    """Selects the appropriate x-column selection from a dataset for ML use."""
    if x is None:
        return df.columns.difference(pd.Index([y]))
    elif isinstance(x, str):
        return pattern(xs, df.columns)
    else:
        return xs


@deprecated("0.2.8", "0.2.9", instead="preprocess_continuous_X_y", reason="reliance on MetaPanda object and pipes")
def ml_ready(df: "MetaPanda",
             x: SelectorType,
             y: Optional[str] = None,
             x_std: bool = True,
             y_std: bool = False,
             std_func: str = "standardize",
             verbose: int = 0):
    """Make sci-kit learn ready datasets from high-level dataset.

    Operations performed:
        1. standardizes float-based columns
        2. drops columns with one unique value in
        3. drops columns of type 'object' - we do not accept string data for most ML models.
        4. TODO: columns of type 'category' are converted to one-hot encode (only X)
        5. Optional standardization of y-column

    Parameters
    ----------
    df : MetaPanda
        The full dataset with missing values
    x : selector
        The selection of x-column inputs
    y : str, optional
        The target column
    x_std: bool, optional
        If True, standardizes float columns in X
    y_std : bool, optional
        If True, standardizes _y.
    std_func : str/function, optional
        Choose from {'standardize', 'power', 'quantile', 'normalize'}
        If 'standardize': performs z-score scaling
        If 'power': uses power transformation
        If 'quantile': Transforms features using quantiles information
        If 'normalize': Scales vectors to unit norm
        Else let it be a sklearn object (sklearn.preprocessing)
    verbose : int, optional
        If > 0, prints info statements out

    Returns
    -------
    _df : pd.DataFrame
        The reduced full dataset
    _x : np.ndarray
        The input matrix X
    _y : np.ndarray, optional
        The target vector y if present
    xcols : pd.Index
        The names of the columns selected for ML
    """
    std_funcs_ = {'standardize': scale, 'power': power_transform,
                  'quantile': quantile_transform, 'normalize': normalize}
    instance_check(y, (type(None), str))
    instance_check(y_std, bool)
    if isinstance(std_func, str):
        belongs(std_func, tuple(std_funcs_.keys()))

    _df = df.copy()
    # 2. eliminate columns with only one unique value in - only for boolean/category options
    elim_cols = _df.view(lambda z: z.nunique() <= 1)
    # 1. standardize float columns only
    std_cols = intersect(_df.view(x), _df.view(float))
    if len(std_cols) > 0 and x_std:
        _df.transform(std_funcs_[std_func], selector=std_cols, whole=True)
    # 5. standardize y if selected
    if y is not None and y_std:
        _df.transform(std_funcs_[std_func], selector=y, whole=True)
    # 3. add 'object columns' into `elim_cols`
    elim_cols = union(elim_cols, _df.view("object"))
    # drop here
    _df.drop(elim_cols)
    # view x columns as pd.Index
    xcols = df.view(x).difference(elim_cols)
    # if we have no y, just prepare for x
    cols = union(xcols, y) if y is not None else xcols
    # reduced subsets and dropna - get DataFrame
    __df = _df[cols].dropna()

    if verbose > 0:
        print("MLReady Chain: [drop nunique==1: k={} -> standardize: k={} -> y_std: {} -> drop: n={}]".format(
            len(elim_cols), len(std_cols), y is not None and y_std, _df.n_ - __df.shape[0]
        ))

    # access x, y
    _x = np.asarray(__df[xcols]).reshape(-1, 1) if len(xcols) == 1 else np.asarray(__df[xcols])
    if y is None:
        return __df, _x, xcols
    else:
        return __df, _x, np.asarray(__df[y]), xcols


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

    return (df[cols]
            .pipe(zscore)
            .pipe(clean1)
            .select_dtypes(exclude=['category', 'object'])
            .dropna())


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
        if isinstance(xcols, str) or (isinstance(xcols, (list, tuple)) and len(xcols) == 1):
            _x = _x.reshape(-1, 1)
        return _x, _y
    else:
        return __data[xcols], __data[ycols]
