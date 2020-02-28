#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit basic machine learning models."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.utils import listify, standardize, instance_check
from turbopanda.dev import cached

from ._clean import ml_ready
from ._default import model_types, param_types
from ._package import find_sklearn_model


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
        _p = [{header: [find_sklearn_model(model)[0]],
               header+"__"+_get_default_param_name(model): _get_default_params(model)} \
              for model in models]
        return _p
    elif isinstance(models, dict):
        def _handle_single_model(name, _val):
            if isinstance(_val, (list, tuple)):
                # if the values are list/tuple, they are parameter names, use defaults
                _p = {header + "__" + _v: _get_default_params(name, _v) for _v in _val}
                _p[header] = listify(find_sklearn_model(name)[0])
                return _p
            elif isinstance(_val, dict):
                _p = {header + "__" + k: v for k, v in _val.items()}
                _p[header] = listify(find_sklearn_model(name)[0])
                return _p

        arg = [_handle_single_model(model_name, val) for model_name, val in models.items()]
        if len(arg) == 1:
            return arg[0]
        else:
            return arg


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
    def _perform_fit(_df: MetaPanda, _x, _y, _k: int, _repeats: int, _models):
        rep = RepeatedKFold(n_splits=_k, n_repeats=_repeats)
        # the header is 'model_est'
        header = "model"
        # any basic regression model
        pipe = Pipeline([(header, LinearRegression())])
        # get paramgrid - the magic happens here!
        pgrid = _make_parameter_grid(_models, header=header)
        # join default grid parameters to given grid_kws
        def_grid_params = {'scoring': 'neg_root_mean_squared_error',
                           'n_jobs': -2, 'verbose': 2, 'return_train_score': True}
        def_grid_params.update(grid_kws)
        # create gridsearch
        gs = GridSearchCV(pipe, param_grid=pgrid, cv=rep, **def_grid_params)
        # make ml ready
        __df, __xnp, __y = ml_ready(_df, _x, _y)
        # fit the grid - expensive.
        gs.fit(__xnp, __y)
        # generate result
        _result = pd.DataFrame(gs.cv_results_)
        # associate model column to respective results
        _result['model'] = _result['param_model'].apply(lambda f: str(f).split("(")[0])
        # set as MetaPanda
        _met_result = MetaPanda(_result)
        # cast down parameter columns to appropriate type
        _met_result.transform(pd.to_numeric, object, errors="ignore")
        return _met_result

    if cache is not None:
        return cached(_perform_fit, cache, verbose, _df=df, _x=x, _y=y, _k=k, _repeats=repeats, _models=models)
    else:
        return _perform_fit(_df=df, _x=x, _y=y, _k=k, _repeats=repeats, _models=models)
