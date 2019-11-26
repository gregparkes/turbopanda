#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:48:57 2019

@author: gparkes

Handles selection of handles.
"""

import numpy as np
import pandas as pd
import re

from .utils import is_boolean_series, chain_intersection, chain_union, \
    convert_boolean, convert_category


__all__ = ["get_selector", "categorize"]


def _accepted_dtypes():
    return [
         float, int, bool, "category", "float", "int", "bool", "boolean",
         np.float64, np.int64, object, np.int32, np.int16, np.int8,
         np.float32, np.float16
    ]


def _convert_mapper():
    return {
         "category": pd.CategoricalDtype,
         "float": np.float64,
         "int": np.int64,
         "bool": np.bool,
         "boolean": np.bool
    }


def _get_selector_item(df, meta, cached, selector, raise_error=False):
    """
    Accepts:
        type [object, int, float, np.float]
        callable (function)
        pd.Index
        str [regex, df.column name, cached name, meta.column name (bool only)]
    """
    if selector is None:
        return pd.Index([], name=df.columns.name)
    if isinstance(selector, pd.Index):
        # check to see if values in selector match df.column names
        return meta.index.intersection(selector)
    elif selector in _accepted_dtypes():
        # if it's a string option, convert to type
        if selector in _convert_mapper():
            selector = _convert_mapper()[selector]
        return df.columns[df.dtypes.eq(selector)]
    # check if selector is a callable object (i.e function)
    elif callable(selector):
        # call the selector, assuming it takes a pandas.DataFrame argument. Must
        # return a boolean Series.
        ser = df.aggregate(selector, axis=0)
        # perform check
        is_boolean_series(ser)
        # check lengths
        not_same = df.columns.symmetric_difference(ser.index)
        # if this exists, append these true cols on
        if not_same.shape[0] > 0:
            ns = pd.concat([pd.Series(True, index=not_same), ser], axis=0)
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
            col_fetch = [c for c in df.columns if re.search(selector, c)]
            if len(col_fetch) > 0:
                return pd.Index(col_fetch, dtype=object,
                                name=df.columns.name, tupleize_cols=False)
            elif raise_error:
                raise ValueError("selector '{}' yielded no matches.".format(selector))
            else:
                return pd.Index([], name=df.columns.name)
        else:
            # we assume it's in the index, and we return it, else allow pandas to raise the error.
            return pd.Index([selector], name=df.columns.name)
    else:
        raise TypeError("selector type '{}' not recognized".format(type(selector)))


def get_selector(df, meta, cached, selector, raise_error=False, select_join="OR"):
    """
    Selector must be a list/tuple of selectors.
    """
    if isinstance(selector, (tuple, list)):
        # iterate over all selector elements and get pd.Index es.
        s_groups = [_get_selector_item(df, meta, cached, s, raise_error) for s in selector]
        if select_join == "AND":
            return chain_intersection(*s_groups)
        elif select_join == "OR":
            return chain_union(*s_groups)
        # by default, use intersection for AND, union for OR
    else:
        # just one item, return asis
        return _get_selector_item(df, meta, cached, selector, raise_error)


def categorize(df, cat_thresh=20, remove_single_col=True):
    """ Applies changes to df by changing the dtypes of the columns. """
    col_renames = {}
    # iterate over all column names.
    for c in df.columns:
        if df[c].dtype in [np.int64, object] and df[c].count()==df.shape[0]:
            un = df[c].unique()
            if un.shape[0] == 1:
                if remove_single_col:
                    df.drop(c, axis=1, inplace=True)
                else:
                    # convert to bool
                    convert_boolean(df, c, col_renames)
                # there is only one value in this column, remove it.
            elif un.shape[0] == 2:
                if np.all(np.isin([0, 1], un)):
                    # use boolean
                    convert_boolean(df, c, col_renames)
                else:
                    # turn into 2-factor categorical if string, etc.
                    convert_category(df, c, un)
            elif un.shape[0] <= cat_thresh:
                # convert to categorical if string, int, etc.
                convert_category(df, c, un)

    # apply all global rename changes to the column.
    df.rename(columns=col_renames, inplace=True)
