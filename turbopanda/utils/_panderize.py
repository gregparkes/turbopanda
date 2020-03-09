#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains decorators to `pandas-ify` the results of functions."""

import pandas as pd
import numpy as np
from typing import Union, Callable


def panderfy(func: Callable) -> Union[pd.Series, pd.DataFrame]:
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