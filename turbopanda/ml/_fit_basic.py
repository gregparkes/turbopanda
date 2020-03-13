#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit basic machine learning models."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

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
              cv: Tuple[int, int] = (5, 10),
              model: str = "LinearRegression",
              cache: Optional[str] = None,
              plot: bool = False,
              verbose: int = 0,
              **model_kws):
    """Performs a rudimentary fit model with no parameter searching.

    This function helps to provide a broad overview of how successful a given model is on the
    inputs of x -> y. `cv` returns scoring and timing metrics, as well as coefficients if available, whereas
    `yp` provides predicted values for each given `y`.

    Parameters
    ----------
    df : MetaPanda
        The main dataset.
    x : list/tuple of str
        A list of selected column names for x or MetaPanda `selector`.
    y : str
        A selected y column.
    cv : int/tuple, optional (5, 10)
        If int: just reflects number of cross-validations
        If Tuple: (cross_validation, n_repeats) `for RepeatedKFold`
    model : str, sklearn model
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
    cv : MetaPanda
        A dataframe result of cross-validated repeats. Can include w_ coefficients.
    yp : pd.Series
        The predictions for each of y

    Notes
    -----
    Shorthand names for the models, i.e `lm` for LinearRegression or `gauss` for a GaussianProcessRegressor, are accepted.

    By default, `fit_basic` uses the root mean squared error (RMSE). There is currently no option to change this.

    By default, this model assumes you are working with a regression problem. Classification compatibility
    will arrive in a later version.

    See Also
    --------
    fit_grid : Performs exhaustive grid search analysis on the models selected.

    References
    ----------
    .. [1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    """
    # checks
    instance_check(df, MetaPanda)
    instance_check(x, (str, list, tuple, pd.Index))
    instance_check(y, str)
    instance_check(cv, (int, tuple))
    instance_check(cache, (type(None), str))
    instance_check(plot, bool)
    instance_check(verbose, int)
    instance_check(model_kws, dict)
    assert is_sklearn_model(model), "model '{}' is not a valid sklearn model."

    if isinstance(cv, tuple):
        k, repeats = cv
    else:
        k, repeats = cv, 1

    lm, pkg_name = find_sklearn_model(model, "regression")
    # assign keywords to lm
    lm.set_params(**model_kws)
    # make data set machine learning ready.
    _df, _x, _y, _xcols = ml_ready(df, x, y)
    if verbose > 0:
        print("full dataset: {}/{} -> ML: {}/{}({},{})".format(df.n_, df.p_, _df.shape[0], _df.shape[1], _x.shape[1], 1))

    # function 1: performing cross-validated fit.
    def _perform_cv_fit(_x: np.ndarray,
                        _xcols: pd.Index,
                        _y: np.ndarray,
                        _k: int,
                        _repeats: int,
                        _lm,
                        package_name: str) -> "MetaPanda":
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
    def _perform_prediction_fit(_df: pd.DataFrame,
                                _x: np.ndarray,
                                _y: np.ndarray,
                                _yn: str,
                                _k: int,
                                _lm) -> pd.Series:
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
