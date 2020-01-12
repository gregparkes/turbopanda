#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renaming the column names, and variants thereof."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from typing import Tuple, Optional
sys.path.append("../")
from .custypes import SelectorType, PandaIndex
from .utils import is_twotuple, belongs, string_replace, dictzip
from ._selection import _selector_group


def _rename_axis(self,
                 old: PandaIndex,
                 new: PandaIndex,
                 axis: int = 1):
    """Renames a given axis with a dict-mapping."""
    if axis == 1:
        self.df_.rename(columns=dictzip(old, new), inplace=True)
        self.meta_.rename(index=dictzip(old, new), inplace=True)
    elif axis == 0:
        self.df_.rename(index=dictzip(old, new), inplace=True)
    else:
        raise ValueError("axis '{}' not recognized".format(axis))


def rename(self,
           ops: Tuple[str, str],
           selector: Tuple[SelectorType, ...] = None,
           axis: int = 1) -> "MetaPanda":
    """Perform a chain of .str.replace operations on `df_.columns`.

    .. note:: strings that are unchanged remain the same (are not NA'd).

    Parameters
    -------
    ops : list of tuple (2,)
        Where the first value of each tuple is the string to find, with its replacement
        At this stage we only accept *direct* replacements. No regex.
        Operations are performed 'in order'.
    selector : None, str, or tuple args, optional
        Contains either custypes.py, meta column names, column names or regex-compliant strings
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

    curr_cols = sel_cols = self._selector_group(selector, axis)
    # performs the replacement operation inplace
    curr_cols = string_replace(curr_cols, ops)
    # rename using mapping
    self._rename_axis(sel_cols, curr_cols, axis)
    return self


def add_prefix(self,
               pref: str,
               selector: Optional[Tuple[SelectorType, ...]] = None) -> "MetaPanda":
    """Add a prefix to all of the columns or selected columns.

    Parameters
    -------
    pref : str
        The prefix to add
    selector : None, str, or tuple args, optional
        Contains either custypes.py, meta column names, column names or regex-compliant strings
        Allows user to specify subset to rename

    Returns
    ------
    self
    """
    sel_cols = self._selector_group(selector)
    # set to df_ and meta_
    self._rename_axis(sel_cols, sel_cols + pref, axis=1)
    return self


def add_suffix(self,
               suf: str,
               selector: Optional[Tuple[SelectorType, ...]] = None) -> "MetaPanda":
    """Add a suffix to all of the columns or selected columns.

    Parameters
    -------
    suf : str
        The prefix to add
    selector : None, str, or tuple args, optional
        Contains either custypes.py, meta column names, column names or regex-compliant strings
        Allows user to specify subset to rename

    Returns
    ------
    self
    """
    sel_cols = self._selector_group(selector)
    # set to df_ and meta_
    self._rename_axis(sel_cols, sel_cols + suf, axis=1)
    return self
