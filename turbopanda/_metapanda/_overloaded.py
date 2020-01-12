#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overloaded head() function for pandas."""
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
from typing import Union
from copy import deepcopy


def head(self,
         k: int = 5) -> pd.DataFrame:
    """Look at the top k rows of the dataset.

    See `pd.DataFrame.head` documentation for details.

    .. warning:: Not affected by `mode_` attribute.

    Parameters
    --------
    k : int, optional
        Must be 0 < k < n.

    Returns
    -------
    ndf : pandas.DataFrame
        First k rows of df_
    """
    return self.df_.head(k)


def dtypes(self,
           grouped: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """Determine the grouped data custypes.py in the dataset.

    .. warning:: Not affected by `mode_` attribute.

    Parameters
    --------
    grouped : bool, optional
        If True, returns the value_counts of each data type, else returns the direct custypes.py.

    Returns
    -------
    true_types : pd.Series/pd.DataFrame
        A series of index (group/name) and value (count/type)
    """
    return self.meta_['e_types'].value_counts() if grouped else self.meta_['e_types']


def copy(self) -> "MetaPanda":
    """Create a copy of this instance.

    .. warning:: Not affected by `mode_` attribute.

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
    return deepcopy(self)