#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attempts to fit basic machine learning models."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline

from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.dev import cached, cached_chunk
from turbopanda.utils import dictchunk, instance_check

from turbopanda.ml._clean import ml_ready
from turbopanda.ml.plot import parameter_tune
from turbopanda.ml._pgrid import make_parameter_grid


def grid(df: MetaPanda,
         x: SelectorType,
         y: str,
         models,
         cv: Tuple[int, int] = (5, 10),
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
    >>> results = turb.ml.grid(df, "x_column", "y_column", ['Ridge'])
    >>> # these results could then be plotted
    >>> turb.ml.parameter_tune(results)

    References
    ----------
    .. [1] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    """
    # checks
    instance_check(df, MetaPanda)
    instance_check(x, (str, list, tuple, pd.Index))
    instance_check(y, str)
    instance_check(cv, (int, tuple))
    instance_check(models, (tuple, list, dict))
    instance_check(cache, (type(None), str))
    instance_check(plot, bool)
    instance_check(chunks, bool)

    if isinstance(cv, tuple):
        k, repeats = cv
    else:
        k, repeats = cv, 1

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
        def_grid_params = {'scoring': 'neg_root_mean_squared_error',
                           'n_jobs': -2, 'verbose': verbose, 'return_train_score': True}
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
        parameter_tune(_cv_results)

    return _cv_results
