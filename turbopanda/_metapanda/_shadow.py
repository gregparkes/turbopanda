#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the shadowing methods in pandas."""

import pandas as pd
from typing import Union
from turbopanda.utils import instance_check

__all__ = ('head', 'dtypes', 'copy')


def head(self, k: int = 5) -> pd.DataFrame:
    """Look at the top k rows of the dataset.

    See `pd.DataFrame.head` documentation for details.

    Parameters
    --------
    k : int, optional
        Must be 0 < k < n.

    Returns
    -------
    ndf : pandas.DataFrame
        First k rows of df_
    """
    instance_check(k, int)
    return self.df_.head(k)


def dtypes(self, grouped: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """Determine the grouped data types in the dataset.

    Parameters
    --------
    grouped : bool, optional
        If True, returns the value_counts of each data type, else returns the direct types.

    Returns
    -------
    true_types : pd.Series/pd.DataFrame
        A series of index (group/name) and value (count/type)
    """
    instance_check(grouped, bool)
    return self.meta_['true_type'].value_counts() if grouped else self.meta_['true_type']


def copy(self) -> "MetaPanda":
    """Create a copy of this instance.

    Raises
    ------
    CopyException
        Module specific errors with copy.deepcopy
    Returns
    -------
    mdf2 : MetaPanda
        A copy of this object
    See Also
    --------
    copy.deepcopy(x) : Return a deep copy of x.
    """
    from copy import deepcopy

    return deepcopy(self)