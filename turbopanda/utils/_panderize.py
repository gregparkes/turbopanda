#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains decorators to `pandas-ify` the results of functions."""

from typing import Callable, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count


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
