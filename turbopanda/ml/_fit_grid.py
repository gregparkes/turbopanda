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
from turbopanda.utils import listify, standardize, instance_check, broadsort, strpattern, dictchunk
from turbopanda.dev import cached, cached_chunk

from ._clean import ml_ready
from ._default import model_types, param_types
from ._package import find_sklearn_model
from ._plot_tune import parameter_tune_plot


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
               header + "__" + _get_default_param_name(model): broadsort(_get_default_params(model))} \
              for model in models]
        return _p
    elif isinstance(models, dict):
        def _handle_single_model(name, _val):
            if isinstance(_val, (list, tuple)):
                # if the values are list/tuple, they are parameter names, use defaults
                _p = {header + "__" + _v: broadsort(_get_default_params(name, _v)) for _v in _val}
                _p[header] = listify(find_sklearn_model(name)[0])
                return _p
            elif isinstance(_val, dict):
                _p = {header + "__" + k: broadsort(list(v)) for k, v in _val.items()}
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
             models,
             k: int = 5,
             repeats: int = 10,
             cache: Optional[str] = None,
             plot: bool = False,
             chunks: bool = False,
             verbose: int = 0,
             **grid_kws) -> "MetaPanda":
    """Performs exhaustive grid search analysis on the models selected.

    This function aims to encapsulate much of the functionality associated around `GridSearchCV` class
    within scikit-learn. With in-built caching options, flexible selection of inputs and outputs with the
    MetaPanda class.

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
        If not None, cache is a filename handle for caching the `cv_results` as a JSON/csv file.
    plot : bool, optional
        If True, produces appropriate plot determining for each parameter.
    chunks : bool, optional
        If True, and if cache is not None: caches the ML gridsearch into equal-sized chunks.
        This saves chunk files which means that if part of the pipeline breaks, you can start from the previous chunk.
    verbose : int, optional
        If > 0, prints out statements depending on level.

    Other Parameters
    ----------------
    grid_kws : dict, optional
        Additional keywords to assign to GridSearchCV.

    Raises
    ------
    TypeError
        If one of the parameters has wrong input type

    Returns
    -------
    cv_results : MetaPanda
        A dataframe result from GridSearchCV detailing iterations and all scores.

    Notes
    -----
    From version 0.2.3 the `chunks` argument allows for fitting by parts. This means that breaks throughout
    a large pipeline will result only in losses up to the previous chunk. Chunk files are saved as
    '%filename_chunk%i.csv' so beware of clashes. Make sure to set `chunks=True` and `cache=str` where the `models` parameter
    is time-expensive.

    By default, `fit_grid` tunes using the root mean squared error (RMSE). There is currently no option to change this.

    By default, this model assumes you are working with a regression problem. Classification compatibility
    will arrive in a later version.

    See Also
    --------
    fit_basic : Performs a rudimentary fit model with no parameter searching.
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified parameter values for an estimator

    Examples
    --------
    To fit a basic grid, say using Ridge Regression we would:
    >>> import turbopanda as turb
    >>> results = turb.ml.fit_grid(df, "x_column", "y_column", ['Ridge'])
    >>> # these results could then be plotted
    >>> turb.ml.parameter_tune_plot(results)

    References
    ----------
    .. [1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
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
    instance_check(chunks, bool)

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
        __df, __xnp, __y, _xcols = ml_ready(_df, _x, _y)
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
        if chunks:
            # if dictionary, we need to split this into 1-sized list/dict blocks.
            values = dictchunk(models, 1) if isinstance(models, dict) else models
            _cv_results = cached_chunk(_perform_fit, "_models", values, cache, verbose, _df=df,
                                       _x=x, _y=y, _k=k, _repeats=repeats, _models=models)
        else:
            _cv_results = cached(_perform_fit, cache, verbose, _df=df, _x=x, _y=y, _k=k, _repeats=repeats,
                                 _models=models)
    else:
        _cv_results = _perform_fit(_df=df, _x=x, _y=y, _k=k, _repeats=repeats, _models=models)

    if plot:
        parameter_tune_plot(_cv_results)

    return _cv_results


def get_best_model(cv_results: MetaPanda,
                   minimize: bool = True):
    """Returns the best model (with correct params) given the cv_results from a `fit_grid` call.

    The idea behind this function is to fetch from the pool of models the best model
    which could be fed directly into `fit_basic` to get the detailed plots.

    Parameters
    ----------
    cv_results : MetaPanda
        A dataframe result from GridSearchCV detailing iterations and all scores.
    minimize : bool
        Determines whether the scoring function is minimized or maximized

    Returns
    -------
    M : sklearn model
        A parameterized sklearn model (unfitted).

    Notes
    -----
    The returned model is not fitted, you will need to do this yourself.

    See Also
    --------
    fit_basic : Performs a rudimentary fit model with no parameter searching
    """
    if minimize:
        select = cv_results.df_['mean_test_score'].idxmin()
    else:
        select = cv_results.df_['mean_test_score'].idxmax()

    M = cv_results.df_.loc[select, 'model']
    # instantiate a model from text M
    inst_M = find_sklearn_model(M)[0]
    # get dict params
    param_columns = strpattern(
        "param_model__", cv_results.df_.loc[select].dropna().index
    )
    # preprocess dict params to eliminate the header for sklearn models
    _old_params = cv_results.df_.loc[select, param_columns]
    _old_params.index = _old_params.index.str.rsplit("__", 1).str[-1]
    params = _old_params.to_dict()
    # iterate through parameters and cast down potential floats to ints
    for k, v in params.items():
        if isinstance(v, float):
            if v.is_integer():
                params[k] = int(v)

    # set parameters in to the model.
    inst_M.set_params(**params)
    return inst_M
