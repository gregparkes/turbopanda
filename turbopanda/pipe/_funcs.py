#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides access to functions which can be directly fed into pandas.DataFrame.pipe."""

import pandas as pd
from typing import Callable, List

from turbopanda.utils import float_to_integer


__all__ = ('all_float_to_int', 'downcast_all', 'all_low_cardinality_to_categorical',
           'clean1')


def _multi_assign(df: pd.DataFrame,
                  transform_fn: Callable[[pd.Series], pd.Series],
                  condition: Callable[[pd.DataFrame], List[str]]) -> pd.DataFrame:
    """Performs a multi-assignment transformation."""
    # creates a copy of the dataframe
    df_to_use = df.copy()

    return (df_to_use.assign(
        **{
            col: transform_fn(df_to_use[col]) for col in condition(df_to_use)
        }
    ))


def all_float_to_int(df):
    """Attempts to cast all float columns into an integer dtype."""
    df_to_use = df.copy()
    condition = lambda x: list(x.select_dtypes(include=["float"]).columns)
    return _multi_assign(df_to_use, float_to_integer, condition)


def downcast_all(df, target_type, initial_type=None):
    """Attempts to downcast all columns in a pandas.DataFrame to reduce memory."""
    if initial_type is None:
        initial_type = target_type

    df_to_use = df.copy()
    transform_fn = lambda x: pd.to_numeric(x, downcast=target_type)
    condition = lambda x: list(x.select_dtypes(include=[initial_type]).columns)

    return _multi_assign(df_to_use, transform_fn, condition)


def all_low_cardinality_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Casts all low cardinality columns to type 'category' """
    df_to_use = df.copy()
    transform_fn = lambda x: x.astype("category")
    n_entre = df_to_use.shape[0]
    # objects = df_to_use.select_dtypes(include=["object"]).nunique()
    condition = lambda x: (
        x.select_dtypes(include=["object"]).nunique()[
            lambda x: x.div(n_entre).lt(0.5)
        ]
    ).index

    return _multi_assign(df_to_use, transform_fn, condition)


def clean1(df: pd.DataFrame) -> pd.DataFrame:
    """A cleaning method for DataFrames in saving memory and dtypes."""
    df_to_use = df.copy()

    cleaned = (
        df_to_use.pipe(all_low_cardinality_to_categorical)
        .pipe(all_float_to_int)
        .pipe(downcast_all, "float")
        .pipe(downcast_all, "integer")
        .pipe(downcast_all, target_type="unsigned", initial_type="integer")
    )

    return cleaned
