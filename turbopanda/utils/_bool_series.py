#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods for doing boolean checks on a `pd.Series`."""

import pandas as pd
from ._typegroups import c_cat, c_float, c_int


__all__ = ('nunique', 'is_possible_category', 'not_column_float',
           'is_column_float', 'is_column_string', 'is_column_int',
           'is_column_object', 'is_missing_values', 'is_n_value_column',
           'is_unique_id', 'is_potential_id', 'is_potential_stacker')


def nunique(ser: pd.Series) -> int:
    """Convert ser to be nunique."""
    return ser.nunique() if not_column_float(ser) else -1


def is_possible_category(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is categorizable."""
    return ser.dtype in c_cat()


def not_column_float(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is not a float column."""
    return ser.dtype not in c_float()


def is_column_float(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is a float column."""
    return ser.dtype in c_float()


def is_column_string(ser: pd.Series) -> bool:
    """Determines whether the column can operate on strings."""
    try:
        ser.dropna().str.contains("xx")
        return True
    except AttributeError:
        return False


def is_column_int(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is a integer column."""
    return ser.dtype in c_int()


def is_column_object(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is a object column."""
    return ser.dtype in (object, pd.CategoricalDtype)


def is_missing_values(ser: pd.Series) -> bool:
    """Determine whether any missing values are present."""
    return ser.count() < ser.shape[0]


def is_n_value_column(ser: pd.Series, n: int = 1) -> bool:
    """Determine whether the number of unique values equals some value n."""
    return nunique(ser) == n


def is_unique_id(ser: pd.Series) -> bool:
    """Determine whether ser is unique."""
    return ser.is_unique if is_column_int(ser) else False


def is_potential_id(ser: pd.Series,
                    thresh: float = 0.5) -> bool:
    """Determine whether ser is a potential ID column."""
    return (ser.unique().shape[0] / ser.shape[0]) > thresh \
        if is_column_string(ser) else False


def is_potential_stacker(ser: pd.Series,
                         regex: str = ";|\t|,|",
                         thresh: float = 0.1) -> bool:
    """Determine whether ser is a stacker-like column."""
    return ser.dropna().str.contains(regex).sum() > thresh \
        if is_column_string(ser) else False
