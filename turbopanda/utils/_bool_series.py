#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods for doing boolean checks on a `pd.Series`."""

import warnings
import numpy as np
import pandas as pd
from typing import Union

from ._typegroups import c_cat

__all__ = ('nunique', 'is_possible_category', 'not_column_float', 'is_column_boolean',
           'is_column_float', 'is_column_string', 'is_column_int', 'is_column_unsigned_int',
           'is_column_object', 'is_column_discrete', 'is_missing_values', 'is_n_value_column',
           'is_unique_id', 'is_potential_id', 'is_potential_stacker',
           'is_dataframe_float')

""" ######################## PANDAS OPERATIONS on SERIES ################################# """


def nunique(ser: pd.Series) -> int:
    """Convert ser to be nunique."""
    return ser.nunique() if not_column_float(ser) else -1


""" Type checks """


def is_possible_category(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is categorizable."""
    return ser.dtype in c_cat()


def not_column_float(ser: Union[pd.Series, np.ndarray]) -> bool:
    """Checks whether the data type in Series is not a float column."""
    return ser.dtype.kind != "f"


def is_column_boolean(ser: Union[pd.Series, np.ndarray]) -> bool:
    """Checks whether the column data type is a boolean, either type bool or np.uint8/16."""
    return (ser.dtype.kind == 'b') or ((ser.dtype.kind == 'u') and (np.unique(ser).size <= 2))


def is_column_float(ser: Union[pd.Series, np.ndarray]) -> bool:
    """Checks whether the data type in Series is a float column."""
    return ser.dtype.kind == "f"


def is_column_string(ser: pd.Series) -> bool:
    """Determines whether the column can operate on strings."""
    try:
        ser.dropna().str.contains("xx")
        return True
    except AttributeError:
        return False


def is_column_int(ser: Union[pd.Series, np.ndarray]) -> bool:
    """Checks whether the data type in Series is a integer column."""
    return (ser.dtype.kind == 'i') or (ser.dtype.kind == 'u')


def is_column_unsigned_int(ser: Union[pd.Series, np.ndarray]) -> bool:
    """Checks whether the data type in Series is an unsigned integer form."""
    return ser.dtype.kind == 'u'


def is_column_object(ser:Union[pd.Series, np.ndarray]) -> bool:
    """Checks whether the data type in Series is a object column."""
    return ser.dtype.kind == 'O'


def is_column_discrete(ser: Union[pd.Series, np.ndarray]) -> bool:
    """Checks whether the data type is a discrete data type.

    Includes warnings if type 'object' or `signed int` type is passed.

    Raises error if a weird type is passed, such as an Interval object.
    """
    if ser.dtype.kind == "u" or ser.dtype.kind == 'b':
        return True
    elif ser.dtype.kind == 'f':
        return False
    elif ser.dtype.kind == 'O':
        nombre = ser.name if isinstance(ser, pd.Series) else "array"
        warnings.warn("column '{}' of type `object` is passed, default classified as NOT discrete.".format(nombre),
                      UserWarning)
        return False
    elif ser.dtype.kind == 'i':
        nombre = ser.name if isinstance(ser, pd.Series) else "array"
        warnings.warn("column '{}' of type `signed int` is passed, default classified AS discrete.".format(nombre),
                      UserWarning)
        return True
    else:
        raise TypeError("type '{}' not recognized for column, needs implementation".format(ser.dtype.kind))


def is_column_numeric(ser: Union[pd.Series, np.ndarray]) -> bool:
    """Checks whether the 1d array is of type float or int."""
    return ser.dtype.kind == 'f' or ser.dtype.kind == 'u' or ser.dtype.kind == 'i'


""" Dataframe checks using MetaPanda """


def is_dataframe_float(df: Union[np.ndarray, pd.DataFrame]) -> bool:
    """Checks whether every column in df is of type float"""
    if isinstance(df, (np.ndarray, pd.Series, pd.Index)):
        return df.dtype.kind == 'f'
    elif isinstance(df, pd.DataFrame):
        return df.dtypes.eq(float).sum() == df.shape[1]
    else:
        raise TypeError("`df` in `is_dataframe_float` must be of type [ndarray, DataFrame]")


"""Other boolean series checks """


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
