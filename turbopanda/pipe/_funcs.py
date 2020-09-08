#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to functions which can be directly fed into pandas.DataFrame.pipe.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from typing import Callable, List, TypeVar, Optional, Union, Tuple
from sklearn.preprocessing import scale, power_transform

from turbopanda.str import pattern, string_replace
from turbopanda.utils import float_to_integer, bounds_check, intersect
from ._conditions import *


__all__ = ('all_float_to_int', 'downcast_all',
           'all_low_cardinality_to_categorical',
           'zscore', 'yeo_johnson', 'clean1', 'clean2',
           'filter_rows_by_column', 'absolute',
           'rename_index', 'rename_columns', 'replace', 'index_name',
           'impute_missing', 'add_suffix', 'add_prefix', 'krange')


def _multi_assign(df: pd.DataFrame,
                  transform_fn: Callable[[pd.Series], pd.Series],
                  condition: Callable[[pd.DataFrame], List[str]]) -> pd.DataFrame:
    """Performs a multi-assignment transformation."""
    # creates a copy of the dataframe
    df_to_use = df.copy()
    cond = condition(df_to_use)
    if len(cond) == 0:
        return df
    else:
        return (df_to_use.assign(
            **{
                col: transform_fn(df_to_use[col]) for col in cond
            }
        ))


def absolute(df: pd.DataFrame, pat: str = None) -> pd.DataFrame:
    """Performs subselected absolute operation on certain columns."""
    condition = lambda x: list(pattern(pat, x, extended_regex=False)) if pat is not None else df.columns.tolist()
    return _multi_assign(df, np.abs, condition)


def all_float_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Attempts to cast all float columns into an integer dtype."""
    return _multi_assign(df.copy(), float_to_integer, select_float)


def downcast_all(df: pd.DataFrame,
                 target_type: TypeVar,
                 initial_type: Optional[TypeVar] = None) -> pd.DataFrame:
    """Attempts to downcast all columns in a pandas.DataFrame to reduce memory."""
    if initial_type is None:
        initial_type = target_type

    df_to_use = df.copy()
    transform_fn = lambda x: pd.to_numeric(x, downcast=target_type)
    condition = lambda x: list(x.select_dtypes(include=[initial_type]).columns)

    return _multi_assign(df_to_use, transform_fn, condition)


def all_low_cardinality_to_categorical(df: pd.DataFrame,
                                       threshold: float = 0.5) -> pd.DataFrame:
    """Casts all low cardinality columns to type 'category' """
    bounds_check(threshold, 0., 1.)

    df_to_use = df.copy()
    transform_fn = lambda x: x.astype("category")
    n_entre = df_to_use.shape[0]
    # check to see that the condition actually has object types to convert.
    if df.select_dtypes(include=['object']).shape[1] == 0:
        return df
    else:
        # objects = df_to_use.select_dtypes(include=["object"]).nunique()
        condition = lambda x: (
            x.select_dtypes(include=["object"]).nunique()[
                lambda y: y.div(n_entre).lt(threshold)
            ]
        ).index

        return _multi_assign(df_to_use, transform_fn, condition)


""" global standardization functions... """


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes using z-score all float-value columns."""
    return _multi_assign(df.copy(), scale, select_float)


def yeo_johnson(df: pd.DataFrame) -> pd.DataFrame:
    """Performs Yeo-Johnson transformation to all float-value columns. """
    # transformation function is sklearn.power_transform
    return _multi_assign(df.copy(), power_transform, select_float)


""" Filtering rows by a selected column value """


def filter_rows_by_column(df: pd.DataFrame,
                          expression: Callable) -> pd.DataFrame:
    """Performs a filter operation using the expression function.

    Expression function must return a boolean-series like object to filter
    rows by.

    Examples
    --------
    >>> import turbopanda as turb
    >>> df.pipe(turb.pipe.filter_rows_by_column, lambda z: z['column'] == 3)
    """

    return df[expression(df)]


""" Column-based functions """


def rename_columns(df: pd.DataFrame, operations) -> pd.DataFrame:
    """Renames the columns of a pandas.DataFrame using the operations"""
    ncol = string_replace(df.columns, operations)
    # return, mapping new names to the cols
    return df.rename(columns=dict(zip(df.columns, ncol)))


def rename_index(df: pd.DataFrame, operations) -> pd.DataFrame:
    """Renames the index of a pandas.DataFrame using ops """
    nind = string_replace(df.index, operations)
    return df.rename(index=dict(zip(df.index, nind)))


def replace(df: pd.DataFrame, column: str, operations) -> pd.DataFrame:
    """Renames a given column with given operations """
    df[column] = string_replace(df[column], operations)
    return df


def add_suffix(df: pd.DataFrame, suf: str, subtype: Optional[TypeVar] = None) -> pd.DataFrame:
    """Adds a suffix to certain column names."""
    _cols = list(df.select_dtypes(include=[subtype]).columns) if subtype is not None else df.columns
    # map back to sufcols to the dataframe
    return df.rename(columns=dict(zip(_cols, map(lambda s: s + suf, _cols))))


def add_prefix(df: pd.DataFrame, pref: str, subtype: Optional[TypeVar] = None) -> pd.DataFrame:
    """Adds a prefix to certain column names"""
    _cols = list(df.select_dtypes(include=[subtype]).columns) if subtype is not None else df.columns
    # map back to sufcols to the dataframe
    return df.rename(columns=dict(zip(_cols, map(lambda s: pref + s, _cols))))


def index_name(df: pd.DataFrame, new_name: str) -> pd.DataFrame:
    """Renaming the 'index' column created after melt(), reset_index() is really annoying.

    e.g df.rename(columns={"index": "new column name"})

    WHY SO VERBOSE?????
    """
    if "index" in df:
        return df.rename(columns={"index": new_name})
    elif 0 in df:
        return df.rename(columns={0: new_name})
    else:
        return df


""" Some global cleaning functions... """


def clean1(df: pd.DataFrame) -> pd.DataFrame:
    """A cleaning method for DataFrames in saving memory and dtypes.

    Performs:
    - all low cardinality to categorical
    - all float to int
    - downcast all float
    - downcast all int
    - downcast all to unsigned, where int
    """
    df_to_use = df.copy()

    cleaned = (
        df_to_use.pipe(all_low_cardinality_to_categorical)
            .pipe(all_float_to_int)
            .pipe(downcast_all, "float")
            .pipe(downcast_all, "integer")
            .pipe(downcast_all, target_type="unsigned", initial_type="integer")
    )

    return cleaned


def clean2(df: pd.DataFrame) -> pd.DataFrame:
    """A cleaning method for DataFrames in saving memory and dtypes, including standardization.

    Performs [in order]:
    - zscore-transformation, if float
    - downcast all float
    - downcast all int
    - downcast all int to unsigned, if possible
    """
    df_to_use = df.copy()

    cleaned = (
        df_to_use.pipe(zscore)
            .pipe(all_low_cardinality_to_categorical)
            .pipe(downcast_all, "float")
            .pipe(downcast_all, "integer")
            .pipe(downcast_all, target_type="unsigned", initial_type="integer")
    )

    return cleaned


""" Imputation functions """


def impute_missing(df: pd.DataFrame,
                   strategy: Union[int, str] = "mean"):
    """Performs simple imputation on every numeric column within a DataFrame.

    Following the strategy, attempts to impute missing values from any
    column that has them.
    """
    df_to_use = df.copy()
    strat = {
        "mean": np.mean, "median": np.median,
    }
    _sf = strat[strategy] if isinstance(strategy, str) else strategy

    transform_fn = lambda x: x.fillna(_sf(x)) if callable(_sf) else x.fillna(_sf)
    condition = lambda x: list(intersect(select_missing_values(x), select_numeric(x)))
    return _multi_assign(df_to_use, transform_fn, condition)


""" Selection functions """


def krange(df: pd.DataFrame, col: str, k: Union[int, Tuple[int, int]]) -> pd.DataFrame:
    """ Fetches the subset of a dataframe with the k min and k largest values of col."""
    if isinstance(k, tuple):
        largest, smallest = k
    else:
        largest, smallest = k, k

    return pd.concat([
        df.nlargest(largest, col), df.nsmallest(smallest, col)
    ])
