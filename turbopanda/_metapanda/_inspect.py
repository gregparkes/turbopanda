#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the inspection of column names in MetaPanda."""

import warnings
import pandas as pd
from typing import Tuple
from ._types import SelectorType
from ._selection import get_selector

""" INSPECTING COLUMNS """

__all__ = ('view', 'search', 'view_not', 'inspect')


def _handle_mode(df, v, mode='view'):
    if mode == 'view' or mode == 'search':
        return df.columns[df.columns.isin(v)]
    elif mode == 'view_not':
        return df.columns[df.columns.isin(df.columns.difference(v))]
    else:
        raise ValueError("mode '{}' not recognized, choose from ['view', 'search', 'view_not']".format(mode))


def inspect(df, meta, selectors, s=None, join_t='union', axis=1, mode='view'):
    """Handles basic selection using the `get_selector`."""
    if s is None:
        return df.columns if axis == 1 else df.index
    elif axis == 1:
        if len(s) <= 1:
            v = get_selector(df, meta, selectors, s[0], raise_error=False, select_join=join_t)
        else:
            v = get_selector(df, meta, selectors, s, raise_error=False, select_join=join_t)
        return _handle_mode(df, v, mode=mode)
    else:
        raise ValueError("cannot use argument [selector] with axis=0, for rows")


def view(self, *selector: SelectorType) -> pd.Index:
    """View a selection of columns in `df_`.

    Select merely returns the columns of interest selected using this selector.
    Selections of columns can be done by:
        type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
        callable (function) that returns [bool list] of length p
        pd.Index
        str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
        list/tuple of the above

    .. note:: `view` *preserves* the order in which columns appear within the DataFrame.
    Parameters
    ----------
    selector : str or tuple args
        See above for what constitutes an *appropriate selector*.
    Warnings
    --------
    UserWarning
        If the selection returned is empty.
    Returns
    ------
    sel : pd.Index
        The list of column names selected, or empty
    See Also
    --------
    view_not : View the non-selected columns in `df_`.
    search : View the intersection of search terms, for columns in `df_`.
    """
    # we do this 'double-drop' to maintain the order of the DataFrame, because of set operations.
    sel = inspect(self.df_, self.meta_, self.selectors_, list(selector), join_t='union', mode='view')
    # re-order selection so as to not disturb the selection of columns, etc. (due to hashing/set operations)
    return sel


def search(self, *selector: SelectorType) -> pd.Index:
    """View the intersection of search terms, for columns in `df_`.

    Select merely returns the columns of interest selected using this selector.
    Selections of columns can be done by:
        type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
        callable (function) that returns [bool list] of length p
        pd.Index
        str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
        list/tuple of the above

    .. note:: `view` *preserves* the order in which columns appear within the DataFrame.
    Parameters
    -------
    selector : str or tuple args
        See above for what constitutes an *appropriate selector*.
    Warnings
    --------
    UserWarning
        If the selection returned is empty.
    Returns
    ------
    sel : pd.Index
        The list of column names selected, or empty
    See Also
    --------
    view_not : Views the non-selected columns in `df_`.
    view : View a selection of columns in `df_`.
    """
    sel = inspect(self.df_, self.meta_, self.selectors_, list(selector), join_t='intersect', mode='search')
    # re-order selection so as to not disturb the selection of columns, etc. (due to hashing/set operations)
    return sel


def view_not(self, *selector: SelectorType) -> pd.Index:
    """View the non-selected columns in `df_`.

    Select merely returns the columns of interest NOT selected using this selector.
    Selections of columns can be done by:
        type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
        callable (function) that returns [bool list] of length p
        pd.Index
        str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
        list/tuple of the above

    .. note:: `view_not` *preserves* the order in which columns appear within the DataFrame.
    Parameters
    ----------
    selector : str or tuple args
        See above for what constitutes an *appropriate selector*.
    Warnings
    --------
    UserWarning
        If the selection returned is empty.
    Returns
    ------
    sel : pd.Index
        The list of column names NOT selected, or empty
    See Also
    --------
    view : View a selection of columns in `df_`.
    search : View the intersection of search terms, for columns in `df_`.
    """
    sel = inspect(self.df_, self.meta_, self.selectors_, list(selector), join_t='union', mode='view_not')
    # re-order selection so as to not disturb the selection of columns, etc. (due to hashing/set operations)
    return sel
