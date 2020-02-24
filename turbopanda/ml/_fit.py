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
from turbopanda.utils import listify, union, standardize, instance_check
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


def _get_default_params(model, param=None, ret="list", with_est=True,
                        header="model"):
    """Given model m, gets the list of primary parameter values."""
    _mt = model_types()
    _pt = param_types()
    # is the model within mt
    if model in _mt.index:
        _param = _mt.loc[model, "Primary Parameter"] if param is None else param
    else:
        raise ValueError("model '{}' not found in selection".format(model))
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
        return _pt.loc["loss", 'Options'].split(", ")

    if _pt.loc[_param, "DataType"] == "int":
        x = x.astype(np.int)

    if ret == "list":
        x = x.tolist()

    if with_est:
        return {header: listify(_find_sklearn_model(model)[0]), header + "__" + _param: x}
    else:
        return {header + "__" + param: x}


def _make_parameter_grid(models, header="model"):
    """
    models can be one of:
        tuple: list of model names, uses default parameters
        dict: key (model name), value tuple (parameter names) / dict: key (parameter name), value (list of values)
    """
    if isinstance(models, (list, tuple)):
        return [_get_default_params(model, header=header) for model in models]
    elif isinstance(models, dict):
        def _handle_single_model(name, _val):
            if isinstance(_val, (list, tuple)):
                # if the values are list/tuple, they are parameter names, use defaults
                args = [_get_default_params(name, _v, header=header) for _v in _val]
                if len(args) == 1:
                    return args[0]
                else:
                    return args
            elif isinstance(_val, dict):
                _p = {header + "__" + k: v for k, v in _val.items()}
                # make as list
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
              verbose: int = 0):
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

    Returns
    -------
    cv : pd.DataFrame
        A dataframe result of cross-validated repeats. Can include w_ coefficients.
    yp : np.ndarray
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

    def _perform_fit(df: MetaPanda, x, y, k: int, repeats: int, model):
        # define sk model.
        rep = RepeatedKFold(n_splits=k, n_repeats=repeats)
        lm, pkg_name = _find_sklearn_model(model)

        # use view to select x columns.
        xcols = df.view(x)
        # get ml ready versions
        _df, _x, _y = ml_ready(df, x, y)

        # cross validate and predict values.
        cv = cross_validate(lm, _x, _y, cv=rep, return_estimator=True, return_train_score=True, n_jobs=-2)
        yp = cross_val_predict(lm, _x, _y, cv=k)

        # extract coefficients.
        coef = _extract_coefficients_from_model(cv, xcols, pkg_name)

        results = pd.DataFrame(cv)
        # add extra columns for utility.
        results['k'] = np.repeat(np.arange(k), repeats)
        # integrate coefficients into cv if present.
        if not isinstance(coef, (list, tuple)):
            # join on coefficients, prefixing with 'w__' for weights.
            results = results.join(coef.add_prefix("w__"))
        # drop 'estimator' objects.
        results.drop('estimator', axis=1, inplace=True)

        _cv = MetaPanda(results)
        _yp = pd.Series(yp, index=_df.index)

        if plot:
            overview_plot(df, x, y, _cv, _yp)

        # make it a metapanda for ease.
        return _cv, _yp

    if cache is not None:
        return cached(_perform_fit, cache, df=df, x=x, y=y, k=k, repeats=repeats, model=model)
    else:
        return _perform_fit(df=df, x=x, y=y, k=k, repeats=repeats, model=model)


def fit_grid(df: MetaPanda,
             x: SelectorType,
             y: str,
             models: Dict[str, Dict],
             k: int = 5,
             repeats: int = 10,
             cache: Optional[str] = None,
             plot: bool = False,
             verbose: int = 0) -> "MetaPanda":
    """Performs exhaustive grid search analysis on the models selected.

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
        # create gridsearch
        gs = GridSearchCV(pipe, param_grid=pgrid, cv=rep, return_train_score=True,
                          scoring="neg_root_mean_squared_error", n_jobs=-2, verbose=2)
        # make ml ready
        _df, _xnp, _y = ml_ready(df, x, y)
        # fit the grid
        gs.fit(_xnp, _y)
        # generate result
        _result = pd.DataFrame(gs.cv_results_)
        # associate model column to respective results
        _result['model'] = _result['param_model'].apply(lambda f: str(f).split("(")[0])
        return MetaPanda(_result)

    if cache is not None:
        return cached(_perform_fit, cache, verbose, df=df, x=x, y=y, k=k, repeats=repeats, models=models)
    else:
        return _perform_fit(df=df, x=x, y=y, k=k, repeats=repeats, models=models)
