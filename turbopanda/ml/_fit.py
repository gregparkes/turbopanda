#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit basic machine learning models."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_validate, RepeatedKFold, GridSearchCV
from sklearn.base import is_classifier, is_regressor
from sklearn.pipeline import Pipeline

from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.utils import listify, union, standardize, instance_check, insert_suffix
from turbopanda.dev import cached
from ._plot import overview_plot
from ._clean import ml_ready
from ._default import model_types, param_types


def _skpackages():
    return sklearn.linear_model, sklearn.tree, sklearn.neighbors, sklearn.ensemble, sklearn.svm


def _find_sklearn_package(name):
    if isinstance(name, str):
        packages = _skpackages()
        for pkg in packages:
            if hasattr(pkg, name):
                return pkg.__name__
    raise TypeError("model '{}' not recognized as scikit-learn model.".format(name))


def _find_sklearn_model(name):
    if isinstance(name, str):
        packages = _skpackages()
        for pkg in packages:
            if hasattr(pkg, name):
                return getattr(pkg, name)(), pkg.__name__
        raise TypeError("model '{}' not recognized as scikit-learn model.".format(name))
    elif is_classifier(name):
        return name, name.__module__.rsplit(".", 1)[0]
    elif is_regressor(name):
        return name, name.__module__.rsplit(".", 1)[0]
    else:
        raise TypeError("model '{}' not recognized as scikit-learn model.".format(name))


def _model_family(models):
    """Given a list of model names, return the sklearn family package name it belongs to."""
    return [_find_sklearn_package(m).split(".")[-1] for m in models]


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


def _get_default_param_name(model):
    """Given model m, get default parameter name."""
    _mt = model_types()
    if model in _mt.index:
        return _mt.loc[model, "Primary Parameter"]
    else:
        raise ValueError("model '{}' not found in selection".format(model))


def _get_default_params(model, param=None, ret="list"):
    """Given model m, gets the list of primary parameter values."""
    _pt = param_types()
    _param = _get_default_param_name(model) if param is None else param
    # is parameter available?
    if _param not in _pt.index:
        raise ValueError("parameter '{}' not found as valid option: {}".format(_param, _pt.index.tolist()))

    if _pt.loc[_param, "Scale"] == 'log':
        x = np.logspace(np.log10(_pt.loc[_param, "Range Min"]),
                        np.log10(_pt.loc[_param, "Range Max"]),
                        int(_pt.loc[_param, "Suggested N"]))
    elif _pt.loc[_param, 'Scale'] == "normal":
        x = np.linspace(_pt.loc[_param, 'Range Min'],
                        _pt.loc[_param, 'Range Max'],
                        int(_pt.loc[_param, 'Suggested N']))
    else:
        return _pt.loc[_param, 'Options'].split(", ")

    if _pt.loc[_param, "DataType"] == "int":
        x = x.astype(np.int)

    if ret == "list":
        x = x.tolist()
    # don't return model, just get the data.
    return x


def _make_parameter_grid(models, header="model"):
    """
    models can be one of:
        tuple: list of model names, uses default parameters
        dict: key (model name), value tuple/list (parameter names) / dict: key (parameter name), value (list of values)
    """
    if isinstance(models, (list, tuple)):
        _p = [{header: [_find_sklearn_model(model)[0]],
               header+"__"+_get_default_param_name(model): _get_default_params(model)} \
              for model in models]
        return _p
    elif isinstance(models, dict):
        def _handle_single_model(name, _val):
            if isinstance(_val, (list, tuple)):
                # if the values are list/tuple, they are parameter names, use defaults
                _p = {header + "__" + _v: _get_default_params(name, _v) for _v in _val}
                _p[header] = listify(_find_sklearn_model(name)[0])
                return _p
            elif isinstance(_val, dict):
                _p = {header + "__" + k: v for k, v in _val.items()}
                _p[header] = listify(_find_sklearn_model(name)[0])
                return _p

        arg = [_handle_single_model(model_name, val) for model_name, val in models.items()]
        if len(arg) == 1:
            return arg[0]
        else:
            return arg


def fit_basic(df: MetaPanda,
              x: SelectorType,
              y: str,
              k: int = 5,
              repeats: int = 100,
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
    """
    # checks
    instance_check(df, MetaPanda)
    instance_check(x, (str, list, tuple, pd.Index))
    instance_check(y, str)
    instance_check(k, int)
    instance_check(repeats, int)
    instance_check(cache, (type(None), str))
    instance_check(plot, bool)
    instance_check(model_kws, dict)

    lm, pkg_name = _find_sklearn_model(model)
    # assign keywords to lm
    lm.set_params(**model_kws)
    # make data set machine learning ready.
    _df, _x, _y = ml_ready(df, x, y)
    xcols = df.view(x)

    # function 1: performing cross-validated fit.
    def _perform_cv_fit(_x, _xcols, _y, _k, _repeats, _lm, package_name):
        # generate repeatedkfold.
        rep = RepeatedKFold(n_splits=_k, n_repeats=_repeats)
        # cv cross-validate and wrap.
        cv = pd.DataFrame(cross_validate(_lm, _x, _y, cv=rep, return_estimator=True, return_train_score=True, n_jobs=-2))
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
            _perform_cv_fit, cache_cv, verbose, _x=_x, _xcols=xcols, _y=_y, _k=k, _repeats=repeats, _lm=lm, package_name=pkg_name
        )
        _yp = cached(
            _perform_prediction_fit, cache_yp, verbose, _df=_df, _x=_x, _y=_y, _yn=y, _k=k, _lm=lm
        )
    else:
        _cv = _perform_cv_fit(_x, xcols, _y, k, repeats, lm, pkg_name)
        _yp = _perform_prediction_fit(_df, _x, _y, y, k, lm)

    if plot:
        overview_plot(df, x, y, _cv, _yp)
    # return both.
    return _cv, _yp


def fit_grid(df: MetaPanda,
             x: SelectorType,
             y: str,
             models: Dict[str, Dict],
             k: int = 5,
             repeats: int = 10,
             cache: Optional[str] = None,
             plot: bool = False,
             verbose: int = 0,
             grid_kws: Dict = {}) -> "MetaPanda":
    """Performs exhaustive grid search analysis on the models selected.

    By default, fit tunes using the root mean squared error (RMSE).

    Parameters
    ----------
    df : MetaPanda
        The main dataset.
    x : list/tuple of str
        A list of selected column names for x or MetaPanda `selector`.
    y : str
        A selected y column.
    models : tuple/dict
        tuple: list of model names, uses default parameters
        dict: key (model name), value tuple (parameter names) / dict: key (parameter name), value (list of values)
    k : int, optional
        The number of cross-fold validations
    repeats : int, optional
        We use RepeatedKFold, so specifying some repeats
    cache : str, optional
        If not None, stores the resulting model parts in JSON and reloads if present.
    plot : bool, optional
        If True, produces appropriate plot determining for each parameter.
    verbose : int, optional
        If > 0, prints out statements depending on level.
    grid_kws : dict, optional
        Additional keywords to assign to GridSearchCV.

    Returns
    -------
    r : MetaPanda
        A dataframe result from GridSearchCV detailing iterations and all scores.
    """
    # checks
    instance_check(df, MetaPanda)
    instance_check(x, (str, list, tuple, pd.Index))
    instance_check(y, str)
    instance_check(k, int)
    instance_check(models, (tuple, list, dict))
    instance_check(repeats, int)
    instance_check(cache, (type(None), str))
    instance_check(plot, bool)

    # do caching
    def _perform_fit(df: MetaPanda, x, y, k: int, repeats: int, models):
        rep = RepeatedKFold(n_splits=k, n_repeats=repeats)
        # the header is 'model_est'
        header = "model"
        # any basic regression model
        pipe = Pipeline([(header, LinearRegression())])
        # get paramgrid - the magic happens here!
        pgrid = _make_parameter_grid(models, header=header)
        # join default grid parameters to given grid_kws
        def_grid_params = {'scoring': 'neg_root_mean_squared_error',
                           'n_jobs': -2, 'verbose': 2, 'return_train_score': True}
        def_grid_params.update(grid_kws)
        # create gridsearch
        gs = GridSearchCV(pipe, param_grid=pgrid, cv=rep, **def_grid_params)
        # make ml ready
        _df, _xnp, _y = ml_ready(df, x, y)
        # fit the grid - expensive.
        gs.fit(_xnp, _y)
        # generate result
        _result = pd.DataFrame(gs.cv_results_)
        # associate model column to respective results
        _result['model'] = _result['param_model'].apply(lambda f: str(f).split("(")[0])
        # set as MetaPanda
        _met_result = MetaPanda(_result)
        # cast down parameter columns to appropriate type
        _met_result.transform(pd.to_numeric, "object", errors="ignore")
        return _met_result

    if cache is not None:
        return cached(_perform_fit, cache, verbose, df=df, x=x, y=y, k=k, repeats=repeats, models=models)
    else:
        return _perform_fit(df=df, x=x, y=y, k=k, repeats=repeats, models=models)
