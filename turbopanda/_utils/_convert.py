#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Converts pandas.Series from one type to another."""
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from ._bool_series import is_n_value_column, is_column_int, \
    _nunique, is_column_object


def integer_to_boolean(ser: pd.Series) -> pd.Series:
    """ Convert an integer series into boolean if possible """
    return ser.astype(np.bool) if \
        (is_column_int(ser) and is_n_value_column(ser, 2)) else ser


def object_to_categorical(ser: pd.Series,
                          order: Optional[Tuple] = None,
                          thresh: int = 30) -> pd.Series:
    """Convert ser to be of type 'category' if possible."""
    # get uniques if possible
    if not is_column_object(ser):
        return ser
    elif 1 < _nunique(ser) < thresh:
        if order is None:
            return ser.astype(pd.CategoricalDtype(np.sort(ser.dropna().unique()), ordered=True))
        else:
            return ser.astype(pd.CategoricalDtype(order, ordered=True))
    else:
        return ser


def boolean_to_integer(ser: pd.Series) -> pd.Series:
    """ Convert a boolean series into an integer if possible """
    return ser.astype(np.uint8) if (ser.dtype == np.bool) else ser
