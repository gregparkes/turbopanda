#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains decorators to `pandas-ify` the results of functions."""

from typing import Callable, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count

from turbopanda._deprecator import deprecated


__all__ = ("panderfy", 'transform_copy', 'series')


def panderfy(func: Callable):
    """A decorator to convert a list-like output into a pandas.Series or pandas.DataFrame."""
    # check it is string
    if not callable(func):
        raise ValueError('function is not callable')

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
    return pd.Series(values, index=index, name=name) if isinstance(name, str) \
        else pd.Series(values, index=index)


@deprecated("0.2.8", "0.3", instead=".utils.umapp")
def lparallel(func: Callable, *args):
    """Performs a parallel list comprehension operation on f(*args)"""
    if len(args) == 0:
        return func()
    elif len(args) == 1:
        n_cpus = cpu_count()-1 if len(args[0]) > cpu_count() else len(args[0])
        # if we have a numpy array, list etc, expand it out
        return Parallel(n_cpus)(delayed(func)(a) for a in args[0])
    else:
        n_cpus = cpu_count() - 1 if len(args) > cpu_count() else len(args)
        return Parallel(n_cpus)(delayed(func)(arg) for arg in args)
