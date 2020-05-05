#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit a basic PCA decomposition to some data."""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List

from turbopanda import vectorize, Param
from turbopanda._metapanda import SelectorType, MetaPanda
from turbopanda.pipe import zscore, clean1
from turbopanda.str import patproduct, common_substrings
from turbopanda.ml.plot import overview_pca
from turbopanda.utils import instance_check, upcast, bounds_check, nonnegative


__all__ = ('pca', 'stratified_pca')


def _create_pca_model(n, sparsity=0., whiten=False):
    from sklearn.decomposition import PCA, SparsePCA

    if np.isclose(sparsity, 0.):
        return PCA(n_components=n, whiten=whiten)
    else:
        # use a sparsePCA model with sparsity as the alpha L1
        return SparsePCA(n_components=n, alpha=sparsity)


@vectorize
def pca(df: Union[np.ndarray, pd.DataFrame, MetaPanda],
        x: Optional[SelectorType] = None,
        preprocess: bool = True,
        refit: bool = False,
        with_transform: bool = False,
        plot: bool = False,
        whiten: bool = False,
        sparsity: float = 0.,
        variance_threshold: float = 0.9,
        plot_kwargs: Optional[Dict] = None):
    """Fits a PCA model to the data set.

    .. note:: Supports vectorization and `Param`. See `turb.vectorize`.

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
    refit : bool, default=False
        If True, a second PCA model is fitted using the 'best'
        proportional variance/AUC which is returned.
    with_transform : bool, default=False
        If True, returns transformed `X` as a second argument.
    plot : bool, default=False
        If True, plots an 'overview' of the PCA result
    whiten : bool, default=False
        When True (False by default) the components_ vectors are multiplied by the square root of n_samples
        and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
        Whitening will remove some information from the transformed signal (the relative variance scales of the
         components) but can sometime improve the predictive accuracy of the downstream estimators
          by making their data respect some hard-wired assumptions.
    sparsity : float, default = 0.0
        If `sparsity` > 0, uses `SparsePCA` algorithm to induce sparse components using
         L1 norm.
    variance_threshold : float, default=0.9
        Determines the threshold of 'cumulative proportional variance'
         to select a refitted model from. Must be 0 <= `variance_threshold` <= N.
    plot_kwargs : dict, optional
        optional arguments to pass to `pca_overview`.

    Returns
    -------
    model : sklearn.decomposition.PCA
        A PCA model
    X_t : np.ndarray/pd.DataFrame, optional
        The transformed input matrix `X`. Returned if `with_transform` is True.
    """

    instance_check(df, (np.ndarray, pd.DataFrame, MetaPanda))
    instance_check((preprocess, plot, whiten, refit, with_transform), bool)
    instance_check(plot_kwargs, (type(None), dict))
    instance_check(sparsity, float)
    bounds_check(variance_threshold, 0., 1.)

    # define our selected columns
    if x is None:
        if not isinstance(df, np.ndarray):
            x = df.columns
        else:
            x = pd.Index(patproduct("X%d", range(df.shape[1])))
    # extract x columns
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

    # determine PCA model
    _model = _create_pca_model(_x.shape[1], sparsity, whiten)
    #  fit the model
    _model.fit(_x)

    if plot:
        if plot_kwargs is None:
            plot_kwargs = {}
        overview_pca(_model, labels=cols,
                     cutoff_selection=variance_threshold,
                     **plot_kwargs)

    # if we refit the model, refit it and return
    if refit:
        # calculate best index (N)
        _ycum = np.cumsum(_model.explained_variance_ratio_)
        new_n = np.where(_ycum > variance_threshold)[0][0] + 1
        # fit a new PCA model.
        _pcan = _create_pca_model(new_n, sparsity, whiten)
        _pcan.fit(_x)
        if with_transform:
            return _pcan, pd.DataFrame(_pcan.transform(_x), index=_x.index)
        else:
            return _pcan
    else:
        if with_transform:
            return _model, pd.DataFrame(_model.transform(_x), index=_x.index)
        else:
            return _model


def stratified_pca(df: Union[np.ndarray, pd.DataFrame, MetaPanda],
                   groups: List[SelectorType],
                   preprocess: bool = True,
                   whiten: bool = False,
                   sparsity: float = 0.,
                   variance_threshold: float = 0.9):
    """Fits a stratified PCA or SparsePCA model to the data set.

    The idea is to break the dataset `df` into `k` groups, perform PCA on each group and join the results
        together.

    Parameters
    ----------
    df : np.ndarray / pd.DataFrame / MetaPanda
        The full dataset
    groups : list of selector
        The different groups to stratify on.
    preprocess : bool, default=True
        Preprocesses the data matrix X if set. Only preprocesses if pandas.DataFrame or above
        Uses the `.pipe.clean1` function which includes zscore,
            dropping object columns and NA.
    whiten : bool, default=False
        When True (False by default) the components_ vectors are multiplied by the square root of n_samples
        and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
        Whitening will remove some information from the transformed signal (the relative variance scales of the
         components) but can sometime improve the predictive accuracy of the downstream estimators
          by making their data respect some hard-wired assumptions.
    sparsity : float, default = 0.0
        If `sparsity` > 0, uses `SparsePCA` algorithm to induce sparse components using
         L1 norm.
    variance_threshold : float, default=0.9
        Determines the threshold of 'cumulative proportional variance'
         to select a refitted model from. Must be 0 <= `variance_threshold` <= N.

    Returns
    -------
    models : list of sklearn.PCA
        Returns the list of refitted models.
    X_t : pd.DataFrame
        The reconstructed transformed X matrix.
    """
    instance_check(df, (np.ndarray, pd.DataFrame, MetaPanda))

    # call pca using PARAM
    results = pca(df, Param(*groups), preprocess=preprocess,
                  whiten=whiten, sparsity=sparsity,
                  refit=True, with_transform=True,
                  variance_threshold=variance_threshold)

    # set column names for each transformed data
    for r, group in zip(results, groups):
        # if the group is a string, do something
        if isinstance(group, str):
            r[1].columns = patproduct("%s_PC%d", [group], range(1,r[1].shape[1]+1))
        elif isinstance(group, (list, tuple, pd.Index, pd.Series)):
            # use str to find common longest substring.
            lcs = common_substrings(group, min_length=3).idxmax()
            r[1].columns = patproduct(lcs + "_PC%d", range(1,r[1].shape[1]+1))

    # join together pandas chunks
    joined_xt = pd.concat(
        list(map(lambda x: x[1], results)), axis=1, sort=False, join="inner"
    )

    return list(map(lambda x: x[0], results)), joined_xt
