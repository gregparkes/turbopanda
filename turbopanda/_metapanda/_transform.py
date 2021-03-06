#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides an interface to transformation operations in Metapanda."""

import itertools as it
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from turbopanda.str import common_substrings
from turbopanda.utils import (
    belongs,
    instance_check,
    is_twotuple,
    listify,
    object_to_categorical,
)
from ._drop_values import drop_columns
from ._inspect import inspect
from ._types import SelectorType


def downcast(self):
    """Downcasts all numeric variables into lower-form variables.

    Particularly useful if you wish to store booleans,
        lower-value integers, unsigned integers and floats.

    Returns
    -------
    self
    """
    # transform by converting float, int64 columns to
    self.transform(
        pd.to_numeric,
        selector=("float64", "int64"),
        errors="ignore",
        downcast="unsigned",
    )
    # convert potential object columns to categorical
    self.transform(object_to_categorical, selector="object", method="transform")
    # make 'boolean' columns uint8
    self.transform(pd.Series.astype, selector="bool", dtype=np.uint8)
    # finally, update the meta
    self.update_meta()
    return self


def transform(
    self,
    func: Callable[[pd.Series], pd.Series],
    selector: Optional[SelectorType] = None,
    method: str = "transform",
    whole: bool = False,
    *args,
    **kwargs
):
    """Perform an inplace transformation to column groups.

    This flexible function provides capacity for
        a wide-range of transformations, including custom transformations.

    .. note:: `func` must be a transforming function,
        i.e one that does not change the shape of `df_`.

    Parameters
    -------
    self
    func : function
        A function taking the `pd.Series` as input and output
        If `whole`, accepts `pd.DataFrame` X, returning `pd.DataFrame` Y
    selector : None, str, or tuple args, optional
        Contains either types, meta column names, column
            names or regex-compliant strings
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
        If False, makes use of `pd.DataFrame.<method>`
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
    transform_k : Performs multiple inplace transformations
        to a group of columns within `df_`.
    aggregate : Perform inplace column-wise aggregations to multiple selectors.

    Examples
    --------
    For instance we may wish to square every float column:
    >>> import turbopanda as turb
    >>> import pandas as pd
    >>> tb = turb.MetaPanda(pd.DataFrame({"x": [1, 2, 3], "y": ['a', 'b', 'c']}))
    >>> tb.transform(lambda ser: ser**2, float)
    """
    belongs(method, ["apply", "transform", "applymap"])
    instance_check(whole, bool)
    instance_check(func, "__call__")
    # perform inplace

    selection = inspect(
        self.df_, self.meta_, self.selectors_, selector, axis=1, mode="view"
    )
    # modify
    if callable(func) and selection.shape[0] > 0:
        if whole:
            self.df_.loc[:, selection] = func(
                self.df_.loc[:, selection], *args, **kwargs
            )
        else:
            self.df_.loc[:, selection] = getattr(self.df_.loc[:, selection], method)(
                func, *args, **kwargs
            )
    return self


def transform_k(self, ops: Tuple[Callable[[pd.Series], pd.Series], SelectorType]):
    """Perform multiple inplace transformations to a group of columns within `df_`.

    Allows a chain of k transformations to be applied, in order.

    Parameters
    -------
    self
    ops : list of 2-tuple
        Containing:
            1. func : A function requiring Series as input and output
            2. selector : Contains either types, meta column names,
                column names or regex-compliant strings. Allows user
                to specify subset to rename

    Raises
    ------
    ValueException
        ops must be a 2-tuple shape

    Returns
    -------
    self

    See Also
    --------
    transform : Performs an inplace transformation to a group
        of columns within the `df_` attribute.
    aggregate : Perform inplace column-wise aggregations to multiple selectors.
    """
    is_twotuple(ops)
    for op in ops:
        self.transform(op[0], op[1])
    return self


def aggregate(
    self,
    func: Union[Callable, str],
    name: Optional[str] = None,
    selector: Optional[SelectorType] = None,
    keep: bool = False,
):
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
    selector : (list of) str or tuple args, optional
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
    transform : Performs an inplace transformation
        to a group of columns within the `df_` attribute.
    transform_k : Perform multiple inplace transformations
        to a group of columns within `df_`.#
    aggregate_k : Perform multiple inplace column-wise
        aggregations to multiple selectors

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

    _selection = inspect(
        self.df_, self.meta_, self.selectors_, selector, axis=1, mode="view"
    )

    if name is None:
        if selector in self.selectors_:
            _name = selector
        else:
            # calculate the best name by common substring matching
            _name = common_substrings(_selection).idxmax()
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


def aggregate_k(
    self,
    func: Union[Callable[[pd.DataFrame], pd.Series], str],
    names: Optional[List[str]] = None,
    selectors: Optional[List[SelectorType]] = None,
    keep: bool = False,
):
    """Perform multiple inplace column-wise aggregations to multiple selectors.

    ..note:: Uses the cached selector names to rename if they are used.

    Parameters
    ----------
    self
    func : str or function
      If function: takes DataFrame, returning Series, for each selection.
      If str: choose from {'mean', 'sum', 'min', 'max', 'std', 'count'}.
    names : str or list of str, optional
      A name for the aggregated column.
      If None, will attempt to extract common pattern subset out of columns.
    selectors : (list of) str or tuple args, optional
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
    transform : Performs an inplace transformation
        to a group of columns within the `df_` attribute.
    transform_k : Perform multiple inplace
        transformations to a group of columns within `df_`.
    aggregate : Perform inplace column-wise
        aggregations using a selector.

    Examples
    --------
    For example if we have a DataFrame such as:
      DF(...,['c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'e1', 'e2', 'e3'])
      We aggregate such that columns ['c1', 'c2', 'c3'] ->
        c, ['d1', 'd2', 'd3'] -> d, ['e1', 'e2', 'e3'] -> e:
      >>> aggregate_k("sum", names=("C","D","E"), selectors=("c[1-3]","d[1-3]","e[1-3]"))
    """
    # checks
    instance_check((names, selectors), (type(None), list, tuple))

    names = listify(names)
    selectors = listify(selectors)

    for n, s in it.zip_longest(names, selectors):
        # call aggregate on each name, selector pair.
        self.aggregate(func, name=n, selector=s, keep=keep)
    return self
