#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The core of the Selector within MetaPanda."""

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# imports
import sys
import re
from typing import Dict, Tuple
from pandas import CategoricalDtype, concat, Index, Series, DataFrame

# locals
sys.path.append("../")
from .utils import boolean_series_check, intersect, union, t_numpy, dictmap, dictzip
from .custypes import SelectorType


def _selector_types() -> Index:
    """Returns the acceptable selector data types that are searched for."""
    return union(
        t_numpy(),
        tuple(map(lambda n: n.__name__, t_numpy())),
        (float, int, bool, object, CategoricalDtype, "object", "category")
    )


def _regex_column(selector: SelectorType,
                  df: pd.DataFrame,
                  raise_error: bool = False):
    """Use a selector to perform a regex search on the columns within df."""
    c_fetch = [c for c in df.columns if re.search(selector, c)]
    if len(c_fetch) > 0:
        return Index(c_fetch, dtype=object,
                     name=df.columns.name, tupleize_cols=False)
    elif raise_error:
        raise ValueError("selector '{}' yielded no matches.".format(selector))
    else:
        return Index([], name=df.columns.name)


def _get_selector_item(self,
                       selector: SelectorType,
                       raise_error: bool = False) -> Index:
    """
    Accepts:
        type [object, int, float, np.float]
        callable (function)
        pd.Index
        str [regex, df.column name, cached name, meta.column name (bool only)]
    """
    if selector is None:
        return Index([], name=self.df_.columns.name)
    if isinstance(selector, Index):
        # check to see if values in selector match df.column names
        return self.meta_.index.intersection(selector)
    elif selector in _selector_types():
        # if it's a string option, convert to type
        dec_map = {
            **{"object": object, "category": CategoricalDtype},
            **dictzip(map(lambda n: n.__name__, t_numpy()), t_numpy())
        }
        if selector in dec_map.keys():
            selector = dec_map[selector]
        return self.df_.columns[self.df_.dtypes.eq(selector)]
    # check if selector is a callable object (i.e function)
    elif callable(selector):
        # call the selector, assuming it takes a pandas.DataFrame argument. Must
        # return a boolean Series.
        ser = self.df_.aggregate(selector, axis=0)
        # perform check
        boolean_series_check(ser)
        # check lengths
        not_same = self.df_.columns.symmetric_difference(ser.index)
        # if this exists, append these true cols on
        if not_same.shape[0] > 0:
            ns = concat([Series(True, index=not_same), ser], axis=0)
            return self.df_.columns[ns]
        else:
            return df.columns[ser]
    elif isinstance(selector, str):
        # check if the key is in the meta_ column names, only if a boolean column
        if (selector in self.meta_) and (self.meta_[selector].dtype == bool):
            return self.df_.columns[self.meta_[selector]]
        elif selector in self.selectors_:
            # recursively go down the stack, and fetch the string selectors from that.
            return get_selector(self.selectors_[selector], raise_error)
        # check if key does not exists in df.columns
        elif selector not in self.df_:
            # try regex
            return _regex_column(selector, self.df_, raise_error)
        else:
            # we assume it's in the index, and we return it, else allow pandas to raise the error.
            return Index([selector], name=df.columns.name)
    else:
        raise TypeError("selector type '{}' not recognized".format(type(selector)))


def _get_selector(self,
                  selector: Tuple[SelectorType, ...],
                  raise_error: bool = False,
                  select_join: str = "OR") -> Index:
    """
    Selector must be a list/tuple of selectors.

    Accepts:
        type [object, int, float, np.float]
        callable (function)
        pd.Index
        str [regex, df.column name, cached name, meta.column name (bool only)]
        list/tuple of the above
    """
    if isinstance(selector, (tuple, list)):
        # iterate over all selector elements and get pd.Index es.
        s_groups = [_get_selector_item(s, raise_error) for s in selector]
        # print(s_groups)
        if select_join == "AND":
            return intersect(*s_groups)
        elif select_join == "OR":
            return union(*s_groups)
        # by default, use intersection for AND, union for OR
    else:
        # just one item, return asis
        return _get_selector_item(selector, raise_error)


def _selector_group(self,
                    s: Tuple[SelectorType, ...],
                    axis: int = 1) -> pd.Index:
    if s is None:
        return self.df_.columns if axis == 1 else self.df_.index
    elif axis == 1:
        if isinstance(s, (tuple, list)):
            return self.view(*s)
        else:
            return self.view(s)
    else:
        raise ValueError("cannot use argument [selector] with axis=0, for rows")


def view(self,
         *selector: SelectorType) -> pd.Index:
    """View a selection of columns in `df_`.

    Select merely returns the columns of interest selected using this selector.
    Selections of columns can be done by:
        type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
        callable (function) that returns [bool list] of length p
        pd.Index
        str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
        list/tuple of the above

    .. warning:: Not affected by `mode_` attribute.

    .. note:: `view` *preserves* the order in which columns appear within the DataFrame.

    .. note:: any numpy data type, like np.float64, np.uint8

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
    view_not : View the non-selected columns in `df_`.
    search : View the intersection of search terms, for columns in `df_`.
    """
    # we do this 'double-drop' to maintain the order of the DataFrame, because of set operations.
    return self.df_.columns.drop(self.view_not(*selector))


def search(self,
           *selector: SelectorType) -> pd.Index:
    """View the intersection of search terms, for columns in `df_`.

    Select merely returns the columns of interest selected using this selector.
    Selections of columns can be done by:
        type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
        callable (function) that returns [bool list] of length p
        pd.Index
        str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
        list/tuple of the above

    .. warning:: Not affected by `mode_` attribute.

    .. note:: `view` *preserves* the order in which columns appear within the DataFrame.

    .. note:: any numpy data type, like np.float64, np.uint8

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
    # we do this 'double-drop' to maintain the order of the DataFrame, because of set operations.
    sel = get_selector(selector, raise_error=False, select_join="AND")
    if (sel.shape[0] == 0) and self._with_warnings:
        warnings.warn("in select: '{}' was empty, no columns selected.".format(selector), UserWarning)
    # re-order selection so as to not disturb the selection of columns, etc. (due to hashing/set operations)
    return sel


def view_not(self,
             *selector: SelectorType) -> pd.Index:
    """View the non-selected columns in `df_`.

    Select merely returns the columns of interest NOT selected using this selector.
    Selections of columns can be done by:
        type [object, int, float, numpy.dtype*, pandas.CategoricalDtype]
        callable (function) that returns [bool list] of length p
        pd.Index
        str [regex, df.column name, cached name, meta.column name (that references a boolean column)]
        list/tuple of the above

    .. warning:: Not affected by `mode_` attribute.

    .. note:: `view_not` *preserves* the order in which columns appear within the DataFrame.

    .. note::  any numpy data type, like np.float64, np.uint8.

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
        The list of column names NOT selected, or empty

    See Also
    --------
    view : View a selection of columns in `df_`.
    search : View the intersection of search terms, for columns in `df_`.
    """
    sel = get_selector(selector, raise_error=False, select_join="OR")
    if (sel.shape[0] == 0) and self._with_warnings:
        warnings.warn("in view: '{}' was empty, no columns selected.".format(selector), UserWarning)
    # re-order selection so as to not disturb the selection of columns, etc. (due to hashing/set operations)
    return self.df_.columns.drop(sel)
