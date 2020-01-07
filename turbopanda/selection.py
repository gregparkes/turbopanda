#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:48:57 2019

@author: gparkes

Handles selection of handles.
"""

import numpy as np
import re
from pandas import CategoricalDtype, concat, Index, Series

from .utils import boolean_series_check, chain_intersection, chain_union


__all__ = ["regex_column", "get_selector", "_type_encoder_map"]


def _numpy_types():
    return [np.int, np.bool, np.float, np.float64, np.float32, np.float16, np.int64,
            np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64]


def _numpy_string_types():
    return [n.__name__ for n in _numpy_types()]


def _extra_types():
    return [float, int, bool, object, CategoricalDtype]


def _extra_string_types():
    return ["object", "category"]


def _type_encoder_map():
    return {
        **{object: "object", CategoricalDtype: "category"},
        **dict(zip(_numpy_types(), _numpy_string_types()))
    }


def _type_decoder_map():
    return {
        **{"object":object, "category":CategoricalDtype},
        **dict(zip(_numpy_string_types(), _numpy_types()))
    }


def _accepted_dtypes():
    return _numpy_types() + _numpy_string_types() + _extra_types() + _extra_string_types()


def regex_column(selector, df, raise_error=False):
    c_fetch = [c for c in df.columns if re.search(selector, c)]
    if len(c_fetch) > 0:
        return Index(c_fetch, dtype=object,
                     name=df.columns.name, tupleize_cols=False)
    elif raise_error:
        raise ValueError("selector '{}' yielded no matches.".format(selector))
    else:
        return Index([], name=df.columns.name)


def _get_selector_item(df, meta, cached, selector, raise_error=False):
    """
    Accepts:
        type [object, int, float, np.float]
        callable (function)
        pd.Index
        str [regex, df.column name, cached name, meta.column name (bool only)]
    """
    if selector is None:
        return Index([], name=df.columns.name)
    if isinstance(selector, Index):
        # check to see if values in selector match df.column names
        return meta.index.intersection(selector)
    elif selector in _accepted_dtypes():
        # if it's a string option, convert to type
        dec_map = _type_decoder_map()
        if selector in dec_map:
            selector = dec_map[selector]
        return df.columns[df.dtypes.eq(selector)]
    # check if selector is a callable object (i.e function)
    elif callable(selector):
        # call the selector, assuming it takes a pandas.DataFrame argument. Must
        # return a boolean Series.
        ser = df.aggregate(selector, axis=0)
        # perform check
        boolean_series_check(ser)
        # check lengths
        not_same = df.columns.symmetric_difference(ser.index)
        # if this exists, append these true cols on
        if not_same.shape[0] > 0:
            ns = concat([Series(True, index=not_same), ser], axis=0)
            return df.columns[ns]
        else:
            return df.columns[ser]
    elif isinstance(selector, str):
        # check if the key is in the meta_ column names, only if a boolean column
        if (selector in meta) and (meta[selector].dtype == bool):
            return df.columns[meta[selector]]
        elif selector in cached:
            # recursively go down the stack, and fetch the string selectors from that.
            return get_selector(df, meta, cached, cached[selector], raise_error)
        # check if key does not exists in df.columns
        elif selector not in df:
            # try regex
            return regex_column(selector, df, raise_error)
        else:
            # we assume it's in the index, and we return it, else allow pandas to raise the error.
            return Index([selector], name=df.columns.name)
    else:
        raise TypeError("selector type '{}' not recognized".format(type(selector)))


def get_selector(df, meta, cached, selector, raise_error=False, select_join="OR"):
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
        if select_join == "AND":
            return chain_intersection(*s_groups)
        elif select_join == "OR":
            return chain_union(*s_groups)
        # by default, use intersection for AND, union for OR
    else:
        # just one item, return asis
        return _get_selector_item(df, meta, cached, selector, raise_error)
