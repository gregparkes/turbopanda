#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to transformation operations in Metapanda."""

import itertools as it
import pandas as pd
from typing import Callable, Optional, Tuple, Union, List

from turbopanda.utils import belongs, instance_check, pairwise, common_substring_match, is_twotuple, listify
from ._types import SelectorType
from ._drop_values import drop_columns
from ._inspect import inspect


def transform(self,
              func: Callable,
              selector: Optional[List[SelectorType]] = None,
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
        Contains either types, meta column names, column names or regex-compliant strings
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

    Raises
    ------
    ValueException
        `method` must be one of {'apply', 'transform', 'applymap'}
    TypeException
        `whole` is not of type bool

    See Also
    --------
    transform_k : Performs multiple inplace transformations to a group of columns within `df_`.
    aggregate : Perform inplace column-wise aggregations to multiple selectors.
    """
    belongs(method, ['apply', 'transform', 'applymap'])
    instance_check(whole, bool)
    instance_check(func, "__call__")
    # perform inplace

    selection = inspect(self.df_, self.meta_, self.selectors_, selector, axis=1, mode='view')
    # modify
    if callable(func) and selection.shape[0] > 0:
        if whole:
            self.df_.loc[:, selection] = func(self.df_.loc[:, selection], *args, **kwargs)
        else:
            self.df_.loc[:, selection] = getattr(self.df_.loc[:, selection], method)(func, *args, **kwargs)
    return self


def transform_k(self, ops: Tuple[Callable, SelectorType]) -> "MetaPanda":
    """Perform multiple inplace transformations to a group of columns within `df_`.

    Allows a chain of k transformations to be applied, in order.

    Parameters
    -------
    ops : list of 2-tuple
        Containing:
            1. func : A function taking the pd.Series x_i as input and returning pd.Series y_i as output
            2. selector : Contains either types, meta column names, column names or regex-compliant strings
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
    aggregate : Perform inplace column-wise aggregations to multiple selectors.
    """
    is_twotuple(ops)
    for op in ops:
        self.transform(op[0], op[1])
    return self


def aggregate(self,
              func: Union[Callable, str],
              name: Optional[str] = None,
              selector: Optional[List[SelectorType]] = None,
              keep: bool = False) -> "MetaPanda":
    """Perform inplace column-wise aggregations using a selector.

    ..note:: Uses the cached selector names to rename if they are used.

    Parameters
    ----------
    func : str or function
        If function: takes a pd.DataFrame x and returns pd.Series y, for each selection.
        If str: choose from {'mean', 'sum', 'min', 'max', 'std', 'count'}.
    name : str, optional
        A name for the aggregated column.
        If None, will attempt to extract common pattern subset out of columns.
    selector : (list of) str or tuple args
        Contains either types, meta column names, column names or regex-compliant strings.
    keep : bool, optional
        If False, drops the rows from which the calculation was made.
        If True, drops the rows from which the calculation was made.

    Returns
    -------
    self

    Raises
    ------
    TypeException
        `name` not of type str or None
        `func` not of callable or str
        `keep` not of type bool

    See Also
    --------
    transform : Performs an inplace transformation to a group of columns within the `df_` attribute.
    transform_k : Perform multiple inplace transformations to a group of columns within `df_`.#
    aggregate_k : Perform multiple inplace column-wise aggregations to multiple selectors

    Examples
    --------
    For example if we have a DataFrame such as:
        DF(...,['c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'e1', 'e2', 'e3'])
        We aggregate such that columns ['c1', 'c2', 'c3'] -> c, etc.
        >>> aggregate("sum", name="C", selector="c[1-3]")
    """
    instance_check(name, (type(None), str))
    instance_check(func, (str, "__call__"))
    instance_check(keep, bool)

    _selection = inspect(self.df_, self.meta_, self.selectors_, selector, axis=1, mode='view')

    if name is None:
        if selector in self.selectors_:
            _name = selector
        else:
            # calculate the best name by common substring matching
            pairs = pairwise(common_substring_match, _selection)
            _name = pd.Series(pairs).value_counts().idxmax()
    else:
        _name = name

    # modify group
    _agg = self.df_[_selection].agg(func, axis=1)
    _agg.name = _name
    # associate with df_, meta_
    if not keep:
        drop_columns(self.df_, self.meta_, _selection)
    # append data to df
    self.df_[_name] = _agg
    return self


def aggregate_k(self,
                func: Union[Callable, str],
                names: Optional[List[str]] = None,
                selectors: List[SelectorType] = None,
                keep: bool = False) -> "MetaPanda":
    """Perform multiple inplace column-wise aggregations to multiple selectors.

    ..note:: Uses the cached selector names to rename if they are used.

    Parameters
    ----------
    func : str or function
      If function: takes a pd.DataFrame x and returns pd.Series y, for each selection.
      If str: choose from {'mean', 'sum', 'min', 'max', 'std', 'count'}.
    names : str, optional
      A name for the aggregated column.
      If None, will attempt to extract common pattern subset out of columns.
    selectors : (list of) str or tuple args
      Contains either types, meta column names, column names or regex-compliant strings.
    keep : bool, optional
      If False, drops the rows from which the calculation was made.
      If True, drops the rows from which the calculation was made.

    Returns
    -------
    self

    Raises
    ------
    TypeException
      `name` not of type str or None
      `func` not of callable or str
      `keep` not of type bool

    See Also
    --------
    transform : Performs an inplace transformation to a group of columns within the `df_` attribute.
    transform_k : Perform multiple inplace transformations to a group of columns within `df_`.
    aggregate : Perform inplace column-wise aggregations using a selector.

    Examples
    --------
    For example if we have a DataFrame such as:
      DF(...,['c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'e1', 'e2', 'e3'])
      We aggregate such that columns ['c1', 'c2', 'c3'] -> c, ['d1', 'd2', 'd3'] -> d, ['e1', 'e2', 'e3'] -> e:
      >>> aggregate_k("sum", names=("C","D","E"), selectors=("c[1-3]","d[1-3]","e[1-3]"))
    """
    # checks
    instance_check(names, (type(None), list, tuple))
    instance_check(selectors, (type(None), list, tuple))

    names = listify(names)
    selectors = listify(selectors)

    for n, s in it.zip_longest(names, selectors):
        # call aggregate on each name, selector pair.
        self.aggregate(func, name=n, selector=s, keep=keep)
    return self


def eval(self, expr: str):
    """Evaluate a Python expression as a string on `df_`.

    See `pandas.eval` documentation for more details.

    TODO: Implement `eval()` function.
        Allows "c = a + b" for single operations; by default keeps a, b; creates c inplace
        Allows "a + b" to return pd.DataFrame of c, not inplace
        Allows regex-style selection of columns to perform multiple evaluations.

    Parameters
    ----------
    expr : str
        The expression to evaluate. This string cannot contain any Python statements, only Python expressions.
        We allow cached 'selectors' to emulate group-like evaluations.

    Returns
    -------
    self

    Examples
    --------
    >>> import turbopanda as turb
    >>> mdf = turb.read("somefile.csv")
    >>> mdf.eval("c=a+b") # creates column c by adding a + b
    """
    return NotImplemented
