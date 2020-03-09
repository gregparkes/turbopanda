#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to cleaning methods ready for ML applications."""

import numpy as np
from typing import Optional

from turbopanda.utils import union, standardize, difference
from turbopanda._metapanda import SelectorType, MetaPanda


def ml_ready(df: MetaPanda, x: SelectorType, y: Optional[str] = None, y_std=False):
    """Make sci-kit learn ready datasets from high-level dataset.

    Operations performed:
        1. standardizes float-based columns
        2. drops columns with one unique value in

    Optional standardization of the y-column.

    Parameters
    ----------
    df : MetaPanda
        The full dataset with missing values
    x : selector
        The selection of x-column inputs
    y : str, optional
        The target column
    y_std : bool, optional
        If True, standardizes _y.

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
    _df = df.copy()
    # eliminate columns with only one unique value in - only for boolean/category options
    elim_cols = _df.view(lambda z: z.nunique() <= 1)
    _df.drop(elim_cols)
    # standardize float columns only
    std_cols = _df.search(x, float)
    if len(std_cols) > 0:
        _df.transform(standardize, selector=std_cols, whole=True)
    if y is not None and y_std:
        _df.transform(standardize, selector=y, whole=True)
    # view x columns as pd.Index
    xcols = df.view(x).difference(elim_cols)
    # if we have no y, just prepare for x
    cols = union(xcols, y) if y is not None else xcols
    # reduced subsets and dropna - get DataFrame
    __df = _df[cols].dropna()
    # access x, y
    _x = np.asarray(__df[xcols]).reshape(-1, 1) if len(xcols) == 1 else np.asarray(__df[xcols])
    if y is None:
        return __df, _x, xcols
    else:
        return __df, _x, np.asarray(__df[y]), xcols
