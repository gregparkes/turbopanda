#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finds duplicated pandas.DataFrame columns.

Inspired from https://hackersandslackers.com/remove-duplicate-columns-in-pandas/
"""
from typing import List
import itertools as it
import pandas as pd

from toolz.itertoolz import sliding_window


def _get_duplicate_cols(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.columns).value_counts()[lambda x: x > 1]


def _get_dup_col_indices(df: pd.DataFrame, col: str) -> List[int]:
    return [x[0] for x in enumerate(df.columns) if x[1]==col][1:]


def _get_all_dup_col_indices(df: pd.DataFrame) -> List[int]:
    dup_cols = _get_duplicate_cols(df).index

    return sorted(list(it.chain.from_iterable(_get_dup_col_indices(df, x) for x in dup_cols)))


def remove_dup_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate columns from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The dataframe to edit.

    Returns
    -------
    ndf : DataFrame
        The un-duplicated column
    """
    indices_to_remove = _get_all_dup_col_indices(df)

    if len(indices_to_remove) == 0:
        return df
    window = list(sliding_window(2, indices_to_remove))

    first = df.iloc[:, :indices_to_remove[0]]
    middle = [df.iloc[:, x[0] + 1 : x[1]] for x in window]
    last = df.iloc[:, indices_to_remove[-1] + 1:]

    if (indices_to_remove[-1]) == (df.shape[1]-1):
        return pd.concat([first] + middle, axis=1, sort=False)
    else:
        return pd.concat([first] + middle + [last], axis=1, sort=False)
