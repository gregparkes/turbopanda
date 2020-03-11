#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to cleaning methods ready for ML applications."""

import numpy as np
import pandas as pd
from typing import Optional

from turbopanda.utils import union, standardize, difference, instance_check
from turbopanda._metapanda import SelectorType, MetaPanda


def ml_ready(df: "MetaPanda",
             x: SelectorType,
             y: Optional[str] = None,
             x_std: bool = True,
             y_std: bool = False,
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

    instance_check(y, (type(None), str))
    instance_check(y_std, bool)

    _df = df.copy()
    # 2. eliminate columns with only one unique value in - only for boolean/category options
    elim_cols = _df.view(lambda z: z.nunique() <= 1)
    # 1. standardize float columns only
    std_cols = _df.search(x, float)
    if len(std_cols) > 0 and x_std:
        _df.transform(standardize, selector=std_cols, whole=True)
    # 5. standardize y if selected
    if y is not None and y_std:
        _df.transform(standardize, selector=y, whole=True)
    # 3. add 'object columns' into `elim_cols`
    elim_cols = union(elim_cols, _df.view("object"))
    # drop here
    _df.drop(elim_cols)
    # 4. perform one-hot encoding of categorical columns

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


def make_polynomial(df: "MetaPanda",
                    x: SelectorType = None,
                    degree: int = 2,
                    **poly_kws):
    """Generates a polynomial full dataset from a smaller subset.

    Parameters
    ----------
    df : MetaPanda
        The full target dataset
    x : selector, optional
        An optional subset of x to perform the polynomialization on
    degree : int
        The degree of the polynomial features

    Other Parameters
    ----------------
    poly_kws : dict
        keywords to pass to `sklearn.preprocessing.PolynomialFeatures()`

    Returns
    -------
    ndf : MetaPanda
        The new target dataset

    See Also
    --------
    sklearn.preprocessing.PolynomialFeatures : Generate polynomial and interaction features
    """
    from sklearn.preprocessing import PolynomialFeatures
    # make a copy
    _xc = df.columns if x is None else x
    # make 'machine learning ready'
    select = df.search(_xc, float)
    _df, _x, _xcol = ml_ready(df, select)
    # define object and fit
    pf = PolynomialFeatures(degree=degree, **poly_kws)
    pf.fit(_x)
    # get new xcols
    _nxcol = [s.replace(" ", "__") for s in pf.get_feature_names(_xcol)]
    # transform x
    __x = pf.transform(_x)
    # rejoin to full dataset and return...
    return MetaPanda(
        pd.DataFrame(
            __x, columns=_nxcol, index=_df.index
        )
    )
