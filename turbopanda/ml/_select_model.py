#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles selecting the best model from a fit.grid call."""

from turbopanda._deprecator import deprecated
from turbopanda.ml._package import find_sklearn_model
from turbopanda.str import pattern


@deprecated("0.2.8", "0.3", reason="This move is to reduce dependency on MetaPanda objects.")
def get_best_model(cv_results: "MetaPanda",
                   minimize: bool = True):
    """Returns the best model (with correct params) given the cv_results from a `fit_grid` call.

    The idea behind this function is to fetch from the pool of models the best model
    which could be fed directly into `fit_basic` to get the detailed plots.

    Parameters
    ----------
    cv_results : MetaPanda
        A dataframe result from `.ml.fit.grid`
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
    param_columns = pattern(
        "param_model__", cv_results.df_.loc[select].dropna().index, False
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
