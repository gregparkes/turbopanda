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


def _get_selector_item(df, meta, cached, selector, raise_error=True):
    if selector in [
            float, int, bool, "category", "float", "int", "bool", "boolean",
            np.float64, np.int64, object
        ]:
        # if it's a string option, convert to type
        string_to_type = {"category": pd.CategoricalDtype, "float": np.float64,
                          "int": np.int64, "bool": np.bool, "boolean": np.bool,
                          float: np.float64, int: np.int64, object:object,
                          np.float64:np.float64, np.int64:np.int64, bool:np.bool}
        nk = string_to_type[selector]
        return df.columns[df.dtypes.eq(nk)]
    # check if selector is a callable object (i.e function)
    elif callable(selector):
        # call the selector, assuming it takes a pandas.DataFrame argument. Must
        # return a boolean Series.
        ser = selector(df)
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
        # check if the key is in the meta_ column names
        if selector in meta:
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
            return selector
    else:
        raise TypeError("selector type '{}' not recognized".format(type(selector)))


def get_selector(df, meta, cached, selector, raise_error=True, select_join="OR"):
    if isinstance(selector, (tuple, list)):
        # iterate over all selector elements and get pd.Index es.
        for s in selector:
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
        if df[c].dtype in [np.int64, object]:
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
