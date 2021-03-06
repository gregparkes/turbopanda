#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit basic machine learning models."""
from __future__ import absolute_import, division, print_function

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize as so

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from turbopanda._deprecator import unimplemented
from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.dev import cached_chunk
from turbopanda.utils import dictchunk, instance_check, bounds_check, nonnegative
from turbopanda.utils import cache as cache_f
from turbopanda.ml._clean import select_xcols, preprocess_continuous_X_y
from turbopanda.ml.plot import parameter_tune
from turbopanda.ml._pgrid import (
    make_parameter_grid,
    make_optimize_grid,
    optimize_grid_for_model,
)
from turbopanda.ml._package import find_sklearn_model, is_sklearn_model


def _min_cross_val_scores(theta, X, y, model, pnames, cv):
    # clone the model
    new_model = clone(model)
    # get the mean scores and use this as the minimization objective
    return -np.mean(
        cross_val_score(
            new_model.set_params(**dict(zip(pnames, theta))),
            X,
            y,
            scoring="neg_root_mean_squared_error",
            cv=cv,
        )
    )


def grid(
    df: Union[pd.DataFrame, "MetaPanda"],
    y: str,
    x: Optional[SelectorType] = None,
    models=("Ridge", "Lasso"),
    cv: Union[int, Tuple[int, int]] = 5,
    cache: Optional[str] = None,
    plot: bool = False,
    chunks: bool = False,
    verbose: int = 0,
    **grid_kws
) -> "MetaPanda":
    """Performs exhaustive grid search analysis on the models selected.

    This function aims to encapsulate much of the functionality associated around `GridSearchCV` class
    within scikit-learn. With in-built caching options, flexible selection of inputs and outputs with the
    MetaPanda class.

    Parameters
    ----------
    df : pd.DataFrame/MetaPanda
        The main dataset.
    y : str
        A selected y column.
    x : list/tuple of str, optional
        A list of selected column names for x or MetaPanda `selector`.
    models : list/dict, default=["Ridge", "Lasso"]
        tuple: list of model names, uses default parameters
        dict: key (model name), value tuple (parameter names) / dict: key (parameter name), value (list of values)
    cv : int/tuple, default=5
        If int: just reflects number of cross-validations
        If Tuple: (cross_validation, n_repeats) `for RepeatedKFold`
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

    Notes ----- From version 0.2.3 the `chunks` argument allows for fitting by parts. This means that breaks
    throughout a large pipeline will result only in losses up to the previous chunk. Chunk files are saved as
    '%filename_chunk%i.csv' so beware of clashes. Make sure to set `chunks=True` and `cache=str` where the `models`
    parameter is time-expensive.

    By default, `grid` tunes using the root mean squared error (RMSE). There is currently no option to change this.

    By default, this model assumes you are working with a regression problem. Classification compatibility
    will arrive in a later version.

    See Also
    --------
    basic : Performs a rudimentary fit model with no parameter searching.
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified parameter values for an estimator

    References
    ----------
    .. [1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    """
    # checks
    instance_check(df, (pd.DataFrame, MetaPanda))
    instance_check(x, (type(None), str, list, tuple, pd.Index))
    instance_check(y, str)
    instance_check(cv, (int, tuple))
    instance_check(cache, (type(None), str))
    instance_check((plot, chunks), bool)
    bounds_check(verbose, 0, 4)

    if is_sklearn_model(models):
        models = [models]
    else:
        if isinstance(models, tuple):
            models = list(models)
        instance_check(models, (list, dict))

    # set dataset if a pandas object
    _df = df.df_ if not isinstance(df, pd.DataFrame) else df
    # retrieve x columns if none
    # set up cv, repeats
    k, repeats = cv if isinstance(cv, tuple) else cv, 1

    # do caching
    def _perform_fit(_df: MetaPanda, _x, _y, _k: int, _repeats: int, _models):
        rep = RepeatedKFold(n_splits=_k, n_repeats=_repeats)
        # the header is 'model_est'
        header = "model"
        # any basic regression model
        pipe = Pipeline([(header, LinearRegression())])
        # get paramgrid - the magic happens here!
        pgrid = make_parameter_grid(_models, header=header)
        # join default grid parameters to given grid_kws
        def_grid_params = {
            "scoring": "neg_root_mean_squared_error",
            "n_jobs": -2,
            "verbose": verbose,
            "return_train_score": True,
        }
        def_grid_params.update(grid_kws)
        # create gridsearch
        gs = GridSearchCV(pipe, param_grid=pgrid, cv=rep, **def_grid_params)
        # make ml ready
        __xnp, __y = preprocess_continuous_X_y(_df, _x, _y)
        # fit the grid - expensive.
        gs.fit(__xnp, __y)
        # generate result
        _result = pd.DataFrame(gs.cv_results_)
        # associate model column to respective results
        _result["model"] = _result["param_model"].apply(lambda f: str(f).split("(")[0])
        # set as MetaPanda
        _met_result = MetaPanda(_result)
        # cast down parameter columns to appropriate type
        _met_result.transform(pd.to_numeric, object, errors="ignore")
        return _met_result

    if cache is not None:
        if chunks:
            # if dictionary, we need to split this into 1-sized list/dict blocks.
            values = dictchunk(models, 1) if isinstance(models, dict) else models
            _cv_results = cached_chunk(
                _perform_fit,
                "_models",
                values,
                False,
                cache,
                verbose,
                _df=_df,
                _x=x,
                _y=y,
                _k=k,
                _repeats=repeats,
                _models=models,
            )
        else:
            _cv_results = cache_f(
                cache,
                _perform_fit,
                _df=_df,
                _x=x,
                _y=y,
                _k=k,
                _repeats=repeats,
                _models=models,
            )
    else:
        _cv_results = _perform_fit(
            _df=_df, _x=x, _y=y, _k=k, _repeats=repeats, _models=models
        )

    if plot:
        parameter_tune(_cv_results)

    return _cv_results


@unimplemented
def optimize(
    df: "MetaPanda", x: SelectorType, y: str, models, cv: int = 5, verbose: int = 0
):
    """Performs optimization grid analysis on the models selected.

    This uses `scipy.optimize` function to minimize continuous parameters, for example `alpha` in a Lasso model.

    .. note:: optimization only works on *continuous* parameters with each model.

    TODO: complete `.ml.fit.optimize` function

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
    cv : int/tuple, optional (5, 10)
        If int: just reflects number of cross-validations
        If Tuple: (cross_validation, n_repeats) `for RepeatedKFold`
    cache : str, optional
        If not None, cache is a filename handle for caching the `cv_results` as a JSON/csv file.
    plot : bool, optional
        If True, produces appropriate plot determining for each parameter.
    chunks : bool, optional
        If True, and if cache is not None: caches the ML gridsearch into equal-sized chunks.
        This saves chunk files which means that if part of the pipeline breaks, you can start from the previous chunk.
    verbose : int, optional
        If > 0, prints out statements depending on level.

    Returns
    -------
    cv_results : MetaPanda
        A dataframe result from GridSearchCV detailing iterations and all scores.

    By default, `optimize` tunes using the root mean squared error (RMSE).
       There is currently no option to change this.

    By default, this model assumes you are working with a regression problem. Classification compatibility
        will arrive in a later version.

    See Also
    --------
    grid : Performs exhaustive grid search analysis on the models selected.
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified parameter values for an estimator

    References
     ----------
    .. [1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    """
    # checks
    instance_check(df, MetaPanda)
    instance_check(x, (str, list, tuple, pd.Index))
    instance_check(y, str)
    nonnegative((cv, verbose), int)
    instance_check(models, (tuple, list, dict))
    bounds_check(verbose, 0, 4)

    _df = df.df_ if not isinstance(df, pd.DataFrame) else df
    _xcols = select_xcols(_df, x, y)
    _xnp, _y = preprocess_continuous_X_y(_df, _xcols, y)

    # define the parameter sets
    param_sets = make_optimize_grid(models)

    for m, params in zip(models, param_sets):
        model = find_sklearn_model(m)[0]
        inits, bounds = optimize_grid_for_model(params)
        # minimize for every i element
        mins = [
            so.minimize(
                _min_cross_val_scores,
                x0=i,
                args=(_xnp, _y, model, params, cv),
                bounds=bounds,
            )
            for i in inits
        ]

    pass
