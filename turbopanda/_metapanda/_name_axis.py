#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the naming of columns/index in MetaPanda."""

import pandas as pd
from typing import List, Tuple, Optional
from ._types import SelectorType, PandaIndex
from ._inspect import inspect
from turbopanda._deprecator import deprecated
from turbopanda.utils import is_twotuple, string_replace, belongs


__all__ = ('rename_axis', 'add_prefix', 'add_suffix')


def _rename_axis(df: pd.DataFrame, meta: pd.DataFrame, old: PandaIndex, new: PandaIndex, axis: int = 1):
    if axis == 1:
        df.rename(columns=dict(zip(old, new)), inplace=True)
        meta.rename(index=dict(zip(old, new)), inplace=True)
    elif axis == 0:
        df.rename(index=dict(zip(old, new)), inplace=True)
    else:
        raise ValueError("axis '{}' not recognized".format(axis))


def rename_axis(self,
                ops: Tuple[str, str],
                selector: Optional[Tuple[SelectorType, ...]] = None,
                axis: int = 1) -> "MetaPanda":
    """Perform a chain of .str.replace operations on one of the axes.

    .. note:: strings that are unchanged remain the same (are not NA'd).

    Parameters
    -------
    ops : list of tuple (2,)
        Where the first value of each tuple is the string to find, with its replacement
        At this stage we only accept *direct* replacements. No regex.
        Operations are performed 'in order'
    selector : None, str, or tuple args, optional
        Contains either types, meta column names, column names or regex-compliant strings
        If None, all column names are subject to potential renaming
    axis : int, optional
        Choose from {1, 0} 1 = columns, 0 = index.

    Returns
    -------
    self
    """
    # check ops is right format
    is_twotuple(ops)
    belongs(axis, [0, 1])

    curr_cols = sel_cols = inspect(self.df_, self.meta_, self.selectors_, selector, axis=axis, mode='view')
    # performs the replacement operation inplace
    curr_cols = string_replace(curr_cols, ops)
    # rename using mapping
    _rename_axis(self.df_, self.meta_, sel_cols, curr_cols, axis=axis)
    return self


def add_prefix(self, pref: str,
               selector: Optional[List[SelectorType]] = None) -> "MetaPanda":
    """Add a prefix to all of the columns or selected columns.

    Parameters
    -------
    pref : str
        The prefix to add
    selector : None, str, or tuple args, optional
        Contains either types, meta column names, column names or regex-compliant strings
        Allows user to specify subset to rename

    Returns
    ------
    self
    """
    sel_cols = inspect(self.df_, self.meta_, self.selectors_, selector, axis=axis, mode='view')
    # set to df_ and meta_
    _rename_axis(self.df_, self.meta_, sel_cols, pref + sel_cols, axis=1)
    return self


def add_suffix(self, suf: str,
               selector: Optional[List[SelectorType]] = None) -> "MetaPanda":
    """Add a suffix to all of the columns or selected columns.

    Parameters
    -------
    suf : str
        The prefix to add
    selector : None, str, or tuple args, optional
        Contains either types, meta column names, column names or regex-compliant strings
        Allows user to specify subset to rename

    Returns
    ------
    self
    """
    sel_cols = inspect(self.df_, self.meta_, self.selectors_, selector, axis=axis, mode='view')
    # set to df_ and meta_
    _rename_axis(self.df_, self.meta_, sel_cols, sel_cols + suf, axis=1)
    return self