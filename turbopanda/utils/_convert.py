#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Converts pandas.Series from one type to another."""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

from ._bool_series import is_n_value_column, is_column_int, \
    nunique, is_column_object

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


__all__ = ('listify', 'integer_to_boolean', 'object_to_categorical', 'boolean_to_integer', 'standardize')


def listify(a):
    """Converts 1-length elements or variables into 1-length lists."""
    if isinstance(a, (list, tuple, pd.Index)) and len(a) > 1:
        return a
    else:
        return [a]


def integer_to_boolean(ser: pd.Series) -> pd.Series:
    """ Convert an integer series into boolean if possible """
    return ser.astype(np.bool) if \
        (is_column_int(ser) and is_n_value_column(ser, 2)) else ser


def object_to_categorical(ser: pd.Series,
                          order: Optional[Tuple] = None,
                          thresh: int = 30) -> pd.Series:
    """Convert ser to be of type 'category' if possible."""
    # get uniques if possible
    if 1 < ser.nunique() < thresh:
        if order is None:
            return ser.astype(pd.CategoricalDtype(ser.dropna().unique(), ordered=False))
        else:
            return ser.astype(pd.CategoricalDtype(order, ordered=True))
    else:
        return ser


def boolean_to_integer(ser: pd.Series) -> pd.Series:
    """ Convert a boolean series into an integer if possible """
    return ser.astype(np.uint8) if (ser.dtype == np.bool) else ser


def standardize(x: ArrayLike) -> ArrayLike:
    """
    Performs z-score standardization on vector x.

    Accepts x as [np.ndarray, pd.Series, pd.DataFrame]
    """
    if isinstance(x, pd.Series):
        return (x - x.mean()) / x.std()
    elif isinstance(x, pd.DataFrame):
        return (x - x.mean(axis=0)) / x.std(axis=0)
    elif isinstance(x, np.ndarray):
        return (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    else:
        raise TypeError("x must be of type [pd.Series, pd.DataFrame, np.ndarray]")
