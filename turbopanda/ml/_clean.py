#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to cleaning methods ready for ML applications."""

import numpy as np
from turbopanda.utils import union, standardize


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


def ml_ready(df, x, y):
    """Given MetaPanda, selector x and str y, return cleaned numpy-array ready versions."""
    # view x columns as pd.Index
    xcols = df.view(x)
    # get union
    cols = union(xcols, y)
    # reduced subsets
    _df = df[cols].dropna()
    # access x, y
    _x = np.asarray(_df[xcols]).reshape(-1, 1) if len(x) == 1 else np.asarray(_df[xcols])
    _y = np.asarray(_df[y])
    # standardize
    _x = standardize(_x)
    return _df, _x, _y


