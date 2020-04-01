#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the inspection of column names in MetaPanda."""

import re
import pandas as pd

from ._selection import get_selector
from ._types import SelectorType

from turbopanda._deprecator import deprecated
from turbopanda.utils import instance_check, intersect, union, join

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
    elif isinstance(s, (list, tuple)) and len(s) == 0:
        return pd.Index([], name="colnames")
    elif axis == 1:
        if isinstance(s, (tuple, list)):
            if len(s) <= 1:
                v = get_selector(df, meta, selectors, s[0], raise_error=False, select_join=join_t)
            else:
                v = get_selector(df, meta, selectors, s, raise_error=False, select_join=join_t)
        else:
            v = get_selector(df, meta, selectors, s, raise_error=False, select_join=join_t)
        return _handle_mode(df, v, mode=mode)
    else:
        raise ValueError("cannot use argument [selector] with axis=0, for rows")


def view(self, *selector: SelectorType, **selector_kwargs) -> pd.Index:
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
    *selector : selector or tuple args
        See above for what constitutes an *appropriate selector*.
    **selector_kwargs : dict keyword args
        Passing cache name (k), selector (v). See above for what constitutes an *appropriate selector*.
        Key arguments passed here ARE CACHED, so it is the same as calling `mdf.cache`.

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
    search : View the intersection of columns, for columns in `df_`.
    select : View a subset of columns using a flexible `eval`-like string.
    """
    _ms = list(selector)
    # if we have any keywords, we'll cache them.
    if len(selector_kwargs) > 0:
        self.cache_k(**selector_kwargs)
        # add 'values' to the selector list
        _ms = join(_ms, selector_kwargs.values())
    # we do this 'double-drop' to maintain the order of the DataFrame, because of set operations.
    sel = inspect(self.df_, self.meta_, self.selectors_, _ms, join_t='union', mode='view')
    # re-order selection so as to not disturb the selection of columns, etc. (due to hashing/set operations)
    return sel


@deprecated("0.2.6", "0.2.8", instead="`MetaPanda.select`", reason="redundancy with view, view_not.")
def search(self, *selector: SelectorType) -> pd.Index:
    """View the intersection of columns in `df_`.

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


def select(self, sc) -> pd.Index:
    """View a subset of columns using a flexible `eval`-like string.

    Select merely returns the columns of interest selected using this selector.
    Selections of columns can be done by:
        type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
        callable (function) that returns [bool list] of length p
        pd.Index
        str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
        list/tuple of the above



    .. note:: We do not currently incorporate the use of brackets.

    Parameters
    ----------
    sc : str-like
        The selection string to find an optimal subset of columns.

    Warnings
    --------
    UserWarning
        If the selection returned is empty.

    Returns
    -------
    sel : pd.Index
        The list of column names NOT selected, or empty

    See Also
    --------
    view : View a selection of columns in `df_`.
    search : View the intersection of search terms, for columns in `df_`.

    Examples
    --------
    You can use string names of types to select columns of a certain type:
    >>> import turbopanda as turb
    >>> import pandas as pd
    >>> mdf = turb.MetaPanda(pd.DataFrame({'a': [1., 2.], 'b': [3, 4]}))
    >>> mdf.select("float")
    Index(['a'], dtype='object', name='colnames')

    Or inverses can also be selected using tilde `~`:
    >>> mdf.select("~float")
    Index(['b'], dtype='object', name='colnames')

    Multiple terms can be joined together, include regex-expressions NOT including `&` or `|`,
    in addition to pre-cached strings:
    >>> mdf.select("")
    """
    instance_check(sc, str)

    terms = [c.strip() for c in re.split("[&|]", sc)]
    operator = re.findall("[&|]", sc)
    if len(terms) < 1:
        return pd.Index([])
    else:
        grp = [self.view_not(t[1:]) if t.startswith("~") else self.view(t) for t in terms]
        full = grp[0]
        for mg, op in zip(grp[1:], operator):
            if op == "&":
                full = intersect(full, mg)
            elif op == "|":
                full = union(full, mg)
        return full
