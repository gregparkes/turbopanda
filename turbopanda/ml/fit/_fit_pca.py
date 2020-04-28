#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit a basic PCA decomposition to some data."""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

from turbopanda._metapanda import SelectorType, MetaPanda
from turbopanda.pipe import zscore, clean1
from turbopanda.str import patproduct
from turbopanda.ml.plot import overview_pca
from turbopanda.utils import instance_check, upcast


def pca(df, x=None, preprocess=True, plot=False, plot_kwargs=None):
    """Fits a PCA model to the data set.

    Parameters
    ----------
    df : np.ndarray / pd.DataFrame / MetaPanda
        The full dataset
    x : selector
        A subset of df to select (if MetaPanda), optionally
    preprocess : bool, default=True
        Preprocesses the data matrix X if set. Only preprocesses if pandas.DataFrame or above
        Uses the `.pipe.clean1` function which includes zscore,
            dropping object columns and NA.
    plot : bool, default=False
        If True, plots an 'overview' of the PCA result
    plot_kwargs : dict, optional
        optional arguments to pass to `pca_overview`.

    Returns
    -------
    model : sklearn.decomposition.PCA
        A PCA model
    """
    from sklearn.decomposition import PCA

    instance_check(df, (np.ndarray, pd.DataFrame, MetaPanda))
    instance_check(plot, bool)
    instance_check(plot_kwargs, (type(None), dict))

    if x is None:
        if not isinstance(df, np.ndarray):
            x = df.columns
        else:
            x = pd.Index(patproduct("X%d", range(df.shape[1])))

    if isinstance(df, MetaPanda):
        cols = df.view(x)
    else:
        cols = x

    # generate ML ready subset
    if preprocess and not isinstance(df, np.ndarray):
        _x = (df[cols]
              .pipe(zscore)
              .pipe(clean1)
              .select_dtypes(exclude=['category', 'object'])
              .dropna())
    else:
        _x = df

    # create and fit a model
    _pca = PCA()
    _pca.fit(_x)

    if plot:
        if plot_kwargs is None:
            plot_kwargs = {}
        overview_pca(_pca, labels=cols, **plot_kwargs)

    return _pca
