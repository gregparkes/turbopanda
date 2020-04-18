#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to the naming of columns/index in MetaPanda."""

from typing import Tuple

import pandas as pd

from turbopanda._deprecator import deprecated
from turbopanda.str import string_replace
from turbopanda.utils import belongs, is_twotuple
from ._inspect import inspect
from ._types import PandaIndex, SelectorType

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
                selector: SelectorType = None,
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


@deprecated("0.2.5", "0.3.0", instead="apply('add_prefix')",
            reason="This pandas replacement is not necessary to warrant a function")
def add_prefix(self, pref: str,
               selector: SelectorType = None) -> "MetaPanda":
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

    See Also
    --------
    add_suffix : Add a suffix to all of the columns or selected columns..
    """
    sel_cols = inspect(self.df_, self.meta_, self.selectors_, selector, axis=1, mode='view')
    # set to df_ and meta_
    _rename_axis(self.df_, self.meta_, sel_cols, pref + sel_cols, axis=1)
    return self


@deprecated("0.2.5", "0.3.0", instead="apply('add_suffix')",
            reason="This pandas replacement is not necessary to warrant a function")
def add_suffix(self, suf: str,
               selector: SelectorType = None) -> "MetaPanda":
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

    See Also
    --------
    add_prefix : Add a prefix to all of the columns or selected columns.
    """
    sel_cols = inspect(self.df_, self.meta_, self.selectors_, selector, axis=1, mode='view')
    # set to df_ and meta_
    _rename_axis(self.df_, self.meta_, sel_cols, sel_cols + suf, axis=1)
    return self