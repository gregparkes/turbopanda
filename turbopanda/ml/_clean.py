#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to cleaning methods ready for ML applications."""

import numpy as np
from turbopanda.utils import union, standardize, difference
from turbopanda._metapanda import SelectorType, MetaPanda


def cleaned_subset(df, x, y):
    """Determines an optimal subset with no missing values.

    Parameters
    ----------
    df : MetaPanda
    x : selector
    y : str

    Returns
    -------
    """
    _x = df.view(x)
    cols = union(_x, [y])
    return df[cols].dropna()


def ml_ready(df: MetaPanda, x: SelectorType, y: str):
    """Given MetaPanda, selector x and str y, return cleaned numpy-array ready versions."""
    # create a df copy
    _df = df.copy()
    # standardize float columns only
    std_cols = _df.search(x, float)
    if len(std_cols) > 0:
        _df.transform(standardize, selector=std_cols, whole=True)
    # eliminate columns with only one unique value in - only for boolean/category options
    elim_cols = _df.view(lambda z: z.nunique() <= 1)
    _df.drop(elim_cols)
    # view x columns as pd.Index
    xcols = difference(df.view(x), elim_cols)
    # get union
    cols = union(xcols, y)
    # reduced subsets and dropna - get DataFrame
    __df = _df[cols].dropna()
    # access x, y
    _x = np.asarray(__df[xcols]).reshape(-1, 1) if len(xcols) == 1 else np.asarray(__df[xcols])
    _y = np.asarray(__df[y])
    return __df, _x, _y, xcols


