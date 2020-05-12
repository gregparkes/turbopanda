#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The core of the Selector within MetaPanda."""

# future imports
from __future__ import absolute_import, division, print_function

import itertools as it
# imports
import re
from typing import Dict, Iterable

from pandas import CategoricalDtype, DataFrame, Index, Series, concat

# locals
from turbopanda.utils import dictzip, \
    difference, intersect, join, t_numpy, union
from turbopanda.str import patcolumnmatch
from ._types import SelectorType

__all__ = ("get_selector", "selector_types")


def selector_types() -> Iterable:
    """Returns the acceptable selector data types that are searched for."""
    return join(
        t_numpy(),
        tuple(map(lambda n: n.__name__, t_numpy())),
        (float, int, bool, object, CategoricalDtype, "object", "category")
    )


def _varstrtypes():
    return list(map(lambda t: t.__name__, t_numpy())) + ['object', 'category']


def _vartypes():
    return [float, int, bool, object, CategoricalDtype] + list(t_numpy())


def _get_selector_item(df: DataFrame,
                       meta: DataFrame,
                       cached: Dict[str, SelectorType],
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
        return Index([], name=df.columns.name)
    elif isinstance(selector, Index):
        # check to see if values in selector match df.column names
        return intersect(df.columns, selector)
    elif selector in selector_types():
        # if it's a string option, convert to type
        dec_map = {"object": object,
                   "category": CategoricalDtype,
                   **dictzip(map(lambda n: n.__name__, t_numpy()), t_numpy())
        }
        if selector in dec_map.keys():
            selector = dec_map[selector]
        return df.columns[df.dtypes.eq(selector)]
    # check if selector is a callable object (i.e function)
    elif callable(selector):
        # call the selector, assuming it takes a pandas.DataFrame argument. Must
        # return a boolean Series.
        ser = df.aggregate(selector, axis=0)
        # check lengths
        not_same = difference(df.columns, ser.index)
        # if this exists, append these true cols on
        if not_same.shape[0] > 0:
            ns = concat([Series(True, index=not_same), ser], axis=0, sort=False)
            return df.columns[ns]
        else:
            return df.columns[ser]
    elif isinstance(selector, str):
        # check if our string contains ~, | or &, and use selector if true
        # check if the key is in the meta_ column names, only if a boolean column
        if (selector in meta) and (meta[selector].dtype == bool):
            # UPDATE: this should fix some merging issues.
            return meta[meta[selector]].index
        elif selector in cached:
            # recursively go down the stack, and fetch the string selectors from that.
            return get_selector(df, meta, cached, cached[selector], raise_error)
        # check if key does not exists in df.columns
        elif selector not in df:
            # try regex
            return patcolumnmatch(selector, df)
        else:
            # we assume it's in the index, and we return it, else allow pandas to raise the error.
            return Index([selector], name=df.columns.name)
    else:
        raise TypeError("selector type '{}' not recognized".format(type(selector)))


def get_selector(df: DataFrame,
                 meta: DataFrame,
                 cached: Dict[str, SelectorType],
                 selector: SelectorType,
                 raise_error: bool = False,
                 select_join: str = "union") -> Index:
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
        s_groups = [_get_selector_item(df, meta, cached, s, raise_error) for s in selector]
        # print(s_groups)
        if select_join == "intersect":
            return intersect(*s_groups)
        elif select_join == "union":
            return union(*s_groups)
        else:
            raise ValueError("join '{}' not recognized, use {}".format(select_join, ('union', 'intersect')))
        # by default, use intersection for AND, union for OR
    else:
        # just one item, return asis
        return _get_selector_item(df, meta, cached, selector, raise_error)
