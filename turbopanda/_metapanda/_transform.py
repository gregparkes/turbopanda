#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformation functions to df_ in MetaPanda."""
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Callable, Optional, Tuple
import sys
sys.path.append("../")
from .custypes import SelectorType
from ._selection import _selector_group
from .utils import boolean_series_check


def filter_rows(self,
                func: Callable,
                selector: Tuple[SelectorType, ...] = None,
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
        Contains either custypes.py, meta column names, column names or regex-compliant strings.
        If None, applies `func` to all columns.
    args : list, optional
        Additional arguments to pass as `func(x, *args)`

    Returns
    -------
    self
    """
    # perform inplace
    selection = self._selector_group(selector, axis=1)
    # modify
    if callable(func) and selection.shape[0] == 1:
        bs = func(self.df_[selection[0]], *args)
    elif callable(func) and selection.shape[0] > 1:
        bs = func(self.df_.loc[:, selection], *args)
    else:
        raise ValueError("parameter '{}' not callable".format(func))
    # check that bs is boolean series
    boolean_series_check(bs)
    self.df_ = self.df_.loc[bs, :]
    return self


def transform(self,
              func: Callable,
              selector: Optional[Tuple[SelectorType, ...]] = None,
              method: str = 'transform',
              whole: bool = False,
              *args,
              **kwargs) -> "MetaPanda":
    """Perform an inplace transformation to a group of columns within the `df_` attribute.

    This flexible function provides capacity for a wide-range of transformations, including custom transformations.

    .. note:: `func` must be a transforming function, i.e one that does not change the shape of `df_`.

    Parameters
    -------
    func : function
        A function taking the `pd.Series` x as input and returning `pd.Series` y as output
        If `whole`, accepts `pd.DataFrame` X, returning `pd.DataFrame` Y
    selector : None, str, or tuple args, optional
        Contains either custypes.py, meta column names, column names or regex-compliant strings
        If None, applies the function to all columns.
    method : str, optional
        Allows the user to specify which underlying DataFrame function to call.
            Choose from {'transform', 'apply', 'applymap'}
            - 'transform': Provides shape guarantees and computationally cheapest.
            - 'apply': more generic function. Can be expensive.
            - 'applymap': applies to every ELEMENT (not axis).
            See `pd.DataFrame` for more details.
    whole : bool, optional
        If True, applies whole function. Often computationally cheaper.
        If False, makes use of `pd.DataFrame.<method>`, see `method` parameter. More flexible.
    args : list, optional
        Additional arguments to pass to function(x, *args)
    kwargs : dict, optional
        Additional arguments to pass to function(x, *args, **kwargs)

    Returns
    -------
    self

    See Also
    --------
    transform_k : Performs multiple inplace transformations to a group of columns within `df_`.
    """
    belongs(method, ['apply', 'transform', 'applymap'])
    # perform inplace
    selection = self._selector_group(selector)
    # modify
    if callable(func) and selection.shape[0] > 0:
        if whole:
            self.df_.loc[:, selection] = func(self.df_.loc[:, selection], *args, **kwargs)
        else:
            self.df_.loc[:, selection] = getattr(self.df_.loc[:, selection], method)(func, *args, **kwargs)
    return self


def transform_k(self,
                ops: Tuple[Callable, SelectorType]) -> "MetaPanda":
    """Perform multiple inplace transformations to a group of columns within `df_`.

    Allows a chain of k transformations to be applied, in order.

    Parameters
    -------
    ops : list of 2-tuple
        Containing:
            1. func : A function taking the pd.Series x_i as input and returning pd.Series y_i as output
            2. selector : Contains either custypes.py, meta column names, column names or regex-compliant strings
                Allows user to specify subset to rename

    Raises
    ------
    ValueException
        ops must be a 2-tuple shape

    Returns
    -------
    self

    See Also
    --------
    transform : Performs an inplace transformation to a group of columns within the `df_` attribute.
    """
    is_twotuple(ops)
    for op in ops:
        self.transform(op[0], op[1])
    return self