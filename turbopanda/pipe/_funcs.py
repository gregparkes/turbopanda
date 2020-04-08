#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to functions which can be directly fed into pandas.DataFrame.pipe.
"""
import numpy as np
import pandas as pd
from typing import Callable, List, TypeVar, Optional
from sklearn.preprocessing import scale, power_transform

from turbopanda.utils import float_to_integer, power_scale
from ._conditions import select_float

__all__ = ('all_float_to_int', 'downcast_all', 'all_low_cardinality_to_categorical',
           'zscore', 'yeo_johnson', 'clean1', 'clean2')


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
