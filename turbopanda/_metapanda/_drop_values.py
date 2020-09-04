#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the dropping rows or columns in Metapanda."""

from typing import Callable

import pandas as pd

from ._inspect import inspect
from ._types import SelectorType

__all__ = ('drop', 'keep', 'drop_columns')


def _remove_unused_categories(meta):
    for col in meta.columns[meta.dtypes == "category"]:
        meta[col].cat.remove_unused_categories(inplace=True)


def drop_columns(df: pd.DataFrame, meta: pd.DataFrame, select: pd.Index):
    """Drops columns from df, and indices from meta, using selection 'select'."""
    if select.size > 0:
        df.drop(select, axis=1, inplace=True)
        meta.drop(select, axis=0, inplace=True)
        # remove any unused categories that might've been dropped
        _remove_unused_categories(meta)


def drop(self, *selector: SelectorType) -> "MetaPanda":
    """Drop the selected columns from `df_`.

    Given a selector or group of selectors, drop all of the columns selected within
    this group, applied to `df_`.

    .. note:: `drop` *preserves* the order in which columns appear within the DataFrame.

    Parameters
    -------
    selector : str or tuple args
        Contains either types, meta column names, column names or regex-compliant strings

    Returns
    -------
    self

    See Also
    --------
    keep : Keeps the selected columns from `df_` only.
    """
    # perform inplace
    drop_columns(self.df_, self.meta_, self.view(*selector))
    return self


def keep(self, *selector: SelectorType) -> "MetaPanda":
    """Keep the selected columns from `df_` only.

    Given a selector or group of selectors, keep all of the columns selected within
    this group, applied to `df_`, dropping all others.

    .. note:: `keep` *preserves* the order in which columns appear within the DataFrame.

    Parameters
    --------
    selector : str or tuple args
        Contains either types, meta column names, column names or regex-compliant strings

    Returns
    -------
    self

    See Also
    --------
    drop : Drops the selected columns from `df_`.
    """
    drop_columns(self.df_, self.meta_, self.view_not(*selector))
    return self
