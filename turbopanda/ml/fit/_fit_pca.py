#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit a basic PCA decomposition to some data."""
from __future__ import absolute_import, division, print_function

from turbopanda._metapanda import SelectorType
from turbopanda.pipe import zscore, clean1
from turbopanda.ml.plot import overview_pca


def pca(df, x=None, plot=False):
    """Fits a PCA model to the data set.

    Parameters
    ----------
    df : MetaPanda
        The full dataset
    x : selector
        A subset of df to select, optionally
    plot : bool
        If True, plots an 'overview' of the PCA result

    Returns
    -------
    model : sklearn.decomposition.PCA
        A PCA model
    """
    from sklearn.decomposition import PCA

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
        overview_pca(_pca, labels=cols)

    return _pca
