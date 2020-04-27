#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit a basic PCA decomposition to some data."""
from __future__ import absolute_import, division, print_function

from turbopanda._metapanda import SelectorType
from turbopanda.pipe import zscore, clean1
from turbopanda.ml.plot import overview_pca
from turbopanda.utils import instance_check


def pca(df, x=None, plot=False, plot_kwargs=None):
    """Fits a PCA model to the data set.

    Parameters
    ----------
    df : MetaPanda
        The full dataset
    x : selector
        A subset of df to select, optionally
    plot : bool
        If True, plots an 'overview' of the PCA result
    plot_kwargs : dict, optional
        optional arguments to pass to `pca_overview`.

    Returns
    -------
    model : sklearn.decomposition.PCA
        A PCA model
    """
    from sklearn.decomposition import PCA

    instance_check(plot, bool)
    instance_check(plot_kwargs, (type(None), dict))

    if x is None:
        x = df.columns

    cols = df.view(x)
    # generate ML ready subset
    _x = (df[cols]
          .pipe(zscore)
          .pipe(clean1)
          .select_dtypes(exclude=['category', 'object'])
          .dropna())

    _pca = PCA(_x.shape[1])
    # fit the model
    _pca.fit(_x)

    if plot:
        if plot_kwargs is None:
            plot_kwargs = {}
        overview_pca(_pca, labels=cols, **plot_kwargs)

    return _pca
