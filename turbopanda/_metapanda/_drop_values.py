#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the dropping rows or columns in Metapanda."""

import pandas as pd
from typing import Callable, Tuple
from ._types import SelectorType
from ._inspect import view_not, inspect
from turbopanda.utils import boolean_series_check


__all__ = ('drop', 'keep', 'filter_rows', 'drop_columns')


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


def filter_rows(self,
                func: Callable,
                selector: SelectorType = None,
                *args) -> "MetaPanda":
    """Filter j rows using boolean-index returned from `function`.

    Given a function, filter out rows that do not meet the functions' criteria.

    .. note:: if `selector` is set, the filtering only factors in these columns.

    Parameters
    --------
    func : function
        A function taking the whole dataset or subset, and returning a boolean
        `pd.Series` with True rows kept and False rows dropped
    selector : str or tuple args, optional
        Contains either types, meta column names, column names or regex-compliant strings.
        If None, applies `func` to all columns.
    args : list, optional
        Additional arguments to pass as `func(x, *args)`

    Returns
    -------
    self
    """
    # perform inplace
    selection = inspect(self.df_, self.meta_, self.selectors_, selector, axis=1, )
    # modify
    if callable(func) and selection.shape[0] == 1:
        bs = func(self.df_[selection[0]], *args)
    elif callable(func) and selection.shape[0] > 1:
        bs = func(self.df_.loc[:, selection], *args)
    else:
        raise ValueError("parameter '{}' not callable".format(func))
    # check that bs is boolean series
    boolean_series_check(bs)
    self._df = self.df_.loc[bs, :]
    return self
