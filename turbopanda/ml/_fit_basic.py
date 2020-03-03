#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit basic machine learning models."""

import numpy as np
import pandas as pd
from typing import Optional, Dict

from sklearn.model_selection import RepeatedKFold, cross_validate, cross_val_predict

from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.dev import cached
from turbopanda.utils import listify, union, insert_suffix, instance_check
from ._package import find_sklearn_model, is_sklearn_model
from ._clean import ml_ready
from ._plot_overview import overview_plot


def _extract_coefficients_from_model(cv, x, pkg_name):
    """accepted packages: linear_model, tree, ensemble, svm."""

    if pkg_name == "sklearn.linear_model" or pkg_name == "sklearn.svm":
        cof = np.vstack([m.coef_ for m in cv['estimator']])
        if cof.shape[-1] == 1:
            cof = cof.flatten()
        res = pd.DataFrame(cof, columns=listify(x))
        res['intercept'] = np.vstack([m.intercept_ for m in cv['estimator']]).flatten()
        res.columns = union(listify(x), ['intercept'])
        return res
    elif pkg_name == "sklearn.tree" or pkg_name == "sklearn.ensemble":
        cof = np.vstack([m.feature_importances_ for m in cv['estimator']])
        if cof.shape[-1] == 1:
            cof = cof.flatten()
        res = pd.DataFrame(cof, columns=listify(x))
        res.columns = pd.Index(listify(x))
        return res
    else:
        return []


def fit_basic(df: MetaPanda,
              x: SelectorType,
              y: str,
              k: int = 5,
              repeats: int = 10,
              model: str = "LinearRegression",
              cache: Optional[str] = None,
              plot: bool = False,
              verbose: int = 0,
              model_kws: Dict = {}):
    """Performs a rudimentary fit model with no parameter searching.

    Parameters
    ----------
    df : MetaPanda
        The main dataset.
    x : list/tuple of str
        A list of selected column names for x or MetaPanda `selector`.
    y : str
        A selected y column.
    k : int, optional
        The number of cross-fold validations
    repeats : int, optional
        We use RepeatedKFold, so specifying some repeats
    model : str, sklearn model
        The name of a scikit-learn model, or the model object itself.
    cache : str, optional
        If not None, stores the resulting model parts in JSON and reloads if present.
    plot : bool, optional
        If True, produces `overview_plot` inplace.
    verbose : int, optional
        If > 0, prints out statements depending on level.
    model_kws : dict, optional
        Keywords to pass to the sklearn model which are not parameterized.

    Returns
    -------
    cv : MetaPanda
        A dataframe result of cross-validated repeats. Can include w_ coefficients.
    yp : pd.Series
        The predictions for each of y

    See Also
    --------
    fit_grid : Performs exhaustive grid search analysis on the models selected
    """
    # checks
    instance_check(df, MetaPanda)
    instance_check(x, (str, list, tuple, pd.Index))
    instance_check(y, str)
    instance_check(k, int)
    instance_check(repeats, int)
    instance_check(cache, (type(None), str))
    instance_check(plot, bool)
    instance_check(verbose, int)
    instance_check(model_kws, dict)
    assert is_sklearn_model(model), "model '{}' is not a valid sklearn model."

    lm, pkg_name = find_sklearn_model(model)
    # assign keywords to lm
    lm.set_params(**model_kws)
    # make data set machine learning ready.
    _df, _x, _y, _xcols = ml_ready(df, x, y)

    # function 1: performing cross-validated fit.
    def _perform_cv_fit(_x, _xcols, _y, _k, _repeats, _lm, package_name):
        # generate repeatedkfold.
        rep = RepeatedKFold(n_splits=_k, n_repeats=_repeats)
        # cv cross-validate and wrap.
        cv = pd.DataFrame(cross_validate(_lm, _x, _y, cv=rep, scoring="neg_root_mean_squared_error",
                                         return_estimator=True, return_train_score=True, n_jobs=-2))
        # append results to cv
        cv['k'] = np.repeat(np.arange(_k), _repeats)
        # extract coefficients
        coef = _extract_coefficients_from_model(cv, _xcols, package_name)
        # integrate coefficients
        if not isinstance(coef, (list, tuple)):
            cv = cv.join(coef.add_prefix("w__"))
        # drop estimator
        cv.drop("estimator", axis=1, inplace=True)
        # wrap as metapanda and return
        return MetaPanda(cv)

    # function 2: performing cross-validated predictions.
    def _perform_prediction_fit(_df, _x, _y, _yn, _k, _lm):
        return pd.Series(cross_val_predict(_lm, _x, _y, cv=_k), index=_df.index).to_frame(_yn)

    if cache is not None:
        cache_cv = insert_suffix(cache, "_cv")
        cache_yp = insert_suffix(cache, "_yp")
        _cv = cached(
            _perform_cv_fit, cache_cv, verbose, _x=_x, _xcols=_xcols, _y=_y, _k=k,
            _repeats=repeats, _lm=lm, package_name=pkg_name
        )
        _yp = cached(
            _perform_prediction_fit, cache_yp, verbose, _df=_df, _x=_x, _y=_y, _yn=y, _k=k, _lm=lm
        )
    else:
        _cv = _perform_cv_fit(_x, _xcols, _y, k, repeats, lm, pkg_name)
        _yp = _perform_prediction_fit(_df, _x, _y, y, k, lm)

    if plot:
        overview_plot(df, x, y, _cv, _yp)
    # return both.
    return _cv, _yp
