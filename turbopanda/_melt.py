#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a cleaner way to melt pandas.DataFrames than is provided by the package as-is.
"""

import numpy as np
import pandas as pd

from turbopanda.str import common_substrings, pattern
from turbopanda.utils import instance_check


def melt(df,
         id_vars=None,
         value_vars=None,
         var_name=None,
         value_name=None,
         index_name="index",
         include_index=True,
         include_regex=True,
         include_question_guess=True):
    """Unpivot a DataFrame from wide format to long format, optionally
    leaving identifier variables set.

    .. note:: Does not accept MultIndex pandas.dataFrames.

    Parameters
    ----------
    df : DataFrame
    id_vars : str, tuple, list or ndarray, optional
        Column(s) to use as identifier variables.
        If None: No identifier columns are used
        If str: uses a regex pattern if `include_regex` is True
    value_vars : str, tuple, list, or ndarray, optional
        Column(s) to unpivot. If not specified, uses all columns that
            are not set as `id_vars`
        If str: uses a regex pattern if `include_regex` is True
    var_name : str, optional
        Name to use for the `variable` column. If None it uses the `strategy`
            variable to find the common substring of the names
    value_name : str, optional
        Name to use for the `value` column. If None it uses the `strategy`
            variable to find the common substring of the names
    index_name : str, default="index"
        A name to give to the index if it doesn't have a name value
    include_index : bool, default=True
        If True, it includes the current index column(s) into the `id_vars`
    include_regex : bool, default=True
        If True, uses regular expressions for `id_vars` and `value_vars`
            if they are `str`
    include_question_guess : bool, default=True
        If True, strategy-generated names have a question mark `?` after them

    Returns
    -------
    dfn : pd.DataFrame
        New melted DataFrame

    See Also
    --------
    pandas.DataFrame.melt
    pandas.DataFrame.pivot_table
    """
    # check inputs
    instance_check(df, pd.DataFrame)
    instance_check((id_vars, value_vars), (type(None), str, list, tuple, np.ndarray, pd.Series, pd.Index))
    instance_check((var_name, value_name, index_name), (type(None), str))
    instance_check((include_regex, include_question_guess, include_index), bool)

    _columns = df.columns.tolist()
    _index = df.index

    # perform regex options for id vars and value vars
    if isinstance(id_vars, str) and include_regex:
        # convert to list
        id_vars = pattern(id_vars, df)
    if isinstance(value_vars, str) and include_regex:
        # convert to list
        value_vars = pattern(value_vars, df)

    if id_vars is None:
        if value_vars is not None:
            id_vars = list(set(_columns) - set(value_vars))
        else:
            id_vars = []
    else:
        id_vars = list(id_vars)

    if value_vars is None:
        if id_vars is not None:
            value_vars = list(set(_columns) - set(id_vars))
        else:
            value_vars = _columns
    else:
        value_vars = list(value_vars)

    # if we include the index, we need to reset it
    if include_index:
        # add in the index cols into the data
        df = df.reset_index().rename(columns={"index": index_name})
        # rename index
        if _index.name is not None:
            id_vars.append(_index.name)
        else:
            id_vars.append(index_name)

    # update var_name
    if var_name is None:
        # use common_substring in the id_vars columns
        valns = common_substrings(value_vars)
        if isinstance(valns, pd.Series) and valns.shape[0] > 0:
            _var_name = valns.idxmax()
            # if we have question guess, add it on
        elif isinstance(valns, str):
            _var_name = valns
        elif df.columns.name != "":
            _var_name = df.columns.name
        else:
            _var_name = "variable"
        if include_question_guess:
            _var_name += "?"
    else:
        _var_name = var_name

    if value_name is None:
        _value_name = "value"
    else:
        _value_name = value_name

    return pd.melt(df, id_vars, value_vars, _var_name, _value_name)
