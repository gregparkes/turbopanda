#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Organization of the column dropping/keeping."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import pandas as pd
sys.path.append("../")
from .custypes import SelectorType
from ._selection import view, view_not


def _remove_unused_categories(meta: pd.DataFrame):
    for col in meta.columns[meta.dtypes == "category"]:
        meta[col].cat.remove_unused_categories(inplace=True)


def _drop_columns(df: pd.DataFrame, meta: pd.DataFrame, select: pd.Index):
    if select.size > 0:
        df.drop(select, axis=1, inplace=True)
        meta.drop(select, axis=0, inplace=True)
        # remove any unused categories that might've been dropped
        _remove_unused_categories(meta)


def drop(self,
         *selector: SelectorType) -> "MetaPanda":
    """Drop the selected columns from `df_`.

    Given a selector or group of selectors, drop all of the columns selected within
    this group, applied to `df_`.

    .. note:: `drop` *preserves* the order in which columns appear within the DataFrame.

    Parameters
    -------
    selector : str or tuple args
        Contains either custypes.py, meta column names, column names or regex-compliant strings

    Returns
    -------
    self

    See Also
    --------
    keep : Keeps the selected columns from `df_` only.
    """
    # perform inplace
    _drop_columns(self.view(*selector))
    return self


def keep(self,
         *selector: SelectorType) -> "MetaPanda":
    """Keep the selected columns from `df_` only.

    Given a selector or group of selectors, keep all of the columns selected within
    this group, applied to `df_`, dropping all others.

    .. note:: `keep` *preserves* the order in which columns appear within the DataFrame.

    Parameters
    --------
    selector : str or tuple args
        Contains either custypes.py, meta column names, column names or regex-compliant strings

    Returns
    -------
    self

    See Also
    --------
    drop : Drops the selected columns from `df_`.
    """
    _drop_columns(self.view_not(*selector))
    return self