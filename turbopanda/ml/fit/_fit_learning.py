#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit a learning curve to a basic model."""
from __future__ import absolute_import, division, print_function

import itertools as it
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, learning_curve, permutation_test_score

from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.utils import instance_check
from turbopanda.ml._clean import ml_ready
from turbopanda.ml._package import find_sklearn_model
from turbopanda.ml.plot._plot_learning import learning_curve


def learning(df: "MetaPanda",
             x: SelectorType,
             y: str,
             train_n: Optional[np.ndarray] = None,
             permute_n: int = 0,
             cv: Tuple[int, int] = (5, 20),
             model: str = "LinearRegression",
             cache: Optional[str] = None,
             plot: bool = False,
             verbose: int = 0,
             **model_kws) -> "MetaPanda":
    """Fits a basic model to generate cross-validated training/test scores for different training set sizes.

    A cross-validation generator splits the whole dataset `k` times in training and test data. Subsets of the training set with
    varying sizes will be used to train the estimator and a score for each training subset size and the test set will be computed.
    Afterwards, the scores will be averaged over all `k` runs for each training subset size.

    Parameters
    ----------
    df : MetaPanda (n_samples, n_features)
        The main dataset.
    x : list/tuple of str/selector
        A list of selected column names for x or MetaPanda `selector`.
    y : str
        A selected y column.
    train_n : array-like, with shape (n_ticks,) dtype float or int, optional
        Relative or absolute numbers of training examples that will be used to generate
        learning curve related data.
        If None: uses `linspace(.1, .9, 8)`
    permute_n : int (default 0)
        The number of times to permute y, if > 0, then does full permutation analysis (making 4th plot)
    cv : int/tuple, optional (5, 10)
        If int: just reflects number of cross-validations
        If Tuple: (cross_validation, n_repeats) `for RepeatedKFold`
    model : str/estimator sklearn model that implements `fit` and `predict` methods
        The name of a scikit-learn model, or the model object itself.
    cache : str, optional
        If not None, stores the resulting model parts in JSON and reloads if present.
    plot : bool, optional
        If True, produces `overview_plot` inplace.
    verbose : int, optional
        If > 0, prints out statements depending on level.

    Other Parameters
    ----------------
    model_kws : dict, optional
        Keywords to pass to the sklearn model which are not parameterized.

    Returns
    -------
    results : MetaPanda (n_ticks, 8)
        The results matrix of mean and std scores
    permute_ : np.ndarray (permute_n,), optional
        The permutation scores associated with the permutation analysis

    Notes
    -----
    Shorthand names for the models, i.e `lm` for LinearRegression or `gauss` for a GaussianProcessRegressor, are accepted.

    By default, `fit_learning` uses the root mean squared error (RMSE). There is currently no option to change this.

    By default, this model assumes you are working with a regression problem. Classification compatibility
    will arrive in a later version.

    `permute_n` is set to 0 by default, if you want a permutation histogram, this value must be > 0.

    See Also
    --------
    fit_basic : Performs a rudimentary fit model with no parameter searching.
    fit_grid : Performs exhaustive grid search analysis on the models selected.

    References
    ----------
    .. [1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    """
    # perform checks
    instance_check(y, str)
    instance_check(train_n, (type(None), list, tuple, np.ndarray))
    instance_check(permute_n, int)
    instance_check(cv, (int, tuple))
    instance_check(cache, (type(None), str))
    instance_check(plot, bool)

    if isinstance(cv, tuple):
        k, repeats = cv
    else:
        k, repeats = cv, 1

    lm, pkg_name = find_sklearn_model(model, "regression")
    # assign keywords to lm
    lm.set_params(**model_kws)
    if train_n is None:
        train_n = np.linspace(.1, .9, 8)
    # ml ready
    _df, _x, _y, _xcols = ml_ready(df, x, y)
    if verbose > 0:
        print(
            "full dataset: {}/{} -> ML: {}/{}({},{})".format(df.n_, df.p_, _df.shape[0], _df.shape[1], _x.shape[1], 1))

    rep = RepeatedKFold(n_splits=k, n_repeats=repeats)
    vars_ = learning_curve(lm, _x, _y, train_sizes=train_n,
                           cv=rep, scoring="neg_root_mean_squared_error",
                           n_jobs=-2, verbose=verbose, return_times=True)
    # permutation analysis if permute_n > 0
    if permute_n > 0:
        perm_score_, perm_scorez_, pval = permutation_test_score(lm, _x, _y, cv=rep, n_permutations=permute_n,
                                                                 scoring="neg_root_mean_squared_error",
                                                                 n_jobs=-2, verbose=verbose)

    # outputs
    output_labels_ = ['train_score', 'test_score', 'fit_time', 'score_time']
    # format as df
    results = pd.DataFrame(
        # stack them together
        np.hstack((
            np.stack([np.mean(vars_[i], axis=1) for i in range(1, 5)], axis=1),
            np.stack([np.std(vars_[i], axis=1) for i in range(1, 5)], axis=1)
        )),
        columns=list(it.chain(map(lambda s: "mean_" + s, output_labels_), map(lambda s: "std_" + s, output_labels_)))
    )
    # add N column
    results['N'] = vars_[0]
    R = MetaPanda(results)
    if plot and permute_n > 0:
        learning_curve(R, perm_scorez_)
    elif plot:
        learning_curve(R)
    # return as MetaPanda
    if permute_n > 0:
        return R, perm_score_, perm_scorez_, pval
    else:
        return R
