#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Converts pandas.Series from one type to another."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from turbopanda._deprecator import deprecated
from ._bool_series import is_column_int, is_n_value_column

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]

__all__ = ('listify', 'switcheroo', 'integer_to_boolean',
           'object_to_categorical', 'boolean_to_integer', 'power_scale',
           'float_to_integer', 'standardize', 'ordinal')


def listify(a):
    """Converts 1-length elements or variables into 1-length lists."""
    if isinstance(a, (list, tuple, pd.Index)):
        return a
    else:
        return [a]


def power_scale(ser: pd.Series, factor: float = 2.) -> pd.Series:
    """Scales every value in ser by factor amount."""
    return np.power(ser, factor)


def ordinal(num: int) -> str:
    """Returns the ordinal number of a given integer, as a string.

    eg. 1 -> 1st, 2 -> 2nd, 3 -> 3rd, etc.

    References
    ----------
    Taken from https://www.pythoncentral.io/validate-python-function-parameters-and-return-types-with-decorators/
    """
    if 10 <= num % 100 < 20:
        return '{0}th'.format(num)
    else:
        ordn = {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
        return '{0}{1}'.format(num, ordn)


def switcheroo(ser: pd.Series) -> pd.Series:
    """Switches a pandas.Series index and values set."""
    return pd.Series(ser.index, index=ser.values)


def integer_to_boolean(ser: pd.Series) -> pd.Series:
    """ Convert an integer series into boolean if possible """
    return ser.astype(np.bool) if \
        (is_column_int(ser) and is_n_value_column(ser, 2)) else ser


def float_to_integer(ser: pd.Series) -> pd.Series:
    """ Convert a float series into an integer one if possible """
    try:
        int_ser = ser.astype(int)
        if (ser == int_ser).all():
            return int_ser
        else:
            return ser
    except ValueError:
        return ser


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


@deprecated("0.2.5", "0.2.7", instead=".pipe.zscore")
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
