#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains decorators to `pandas-ify` the results of functions."""

from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count

__all__ = ("panderfy", "transform_copy", "series")


def panderfy(func: Callable):
    """A decorator to convert a list-like output into pandas object."""
    # check it is string
    if not callable(func):
        raise ValueError("function is not callable")

    def _wrapper(*args, **kwargs):
        # call function
        result = func(*args, **kwargs)
        # if list/tuple of something, convert to pandas.series

        if isinstance(result, (list, tuple)):
            return pd.Series(result)
        elif isinstance(result, np.ndarray) and result.ndim == 1:
            return pd.Series(result)
        elif isinstance(result, np.ndarray) and result.ndim == 2:
            return pd.DataFrame(result)
        else:
            return result

    return _wrapper


def transform_copy(old, new):
    """Given an 'old' pandas.Series, pandas.Index or pandas.DataFrame, copy over metadata to
    a 'new' one.
    """
    if isinstance(old, pd.Series):
        return pd.Series(new, index=old.index, name=old.name)
    elif isinstance(old, pd.Index):
        return pd.Index(new, name=old.name)
    elif isinstance(old, pd.DataFrame):
        return pd.DataFrame(new, columns=old.columns, index=old.index)
    else:
        return new


def series(values, index, name=None):
    """Creates a pandas.Series from some values, index and an optional name. Mappable."""
    return (
        pd.Series(values, index=index, name=name)
        if isinstance(name, str)
        else pd.Series(values, index=index)
    )
