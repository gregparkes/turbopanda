#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:13:38 2019

@author: gparkes
"""

import numpy as np
import pandas as pd
import itertools as it
from pandas.api.types import CategoricalDtype



__all__ = ["is_twotuple","instance_check",
           "chain_intersection","chain_union",
           "is_boolean_series","attach_name",
            "check_list_type","not_column_float",
            "is_column_float","is_column_object",
            "convert_category", "convert_boolean",
            "calc_mem"]


def is_twotuple(L):
    """
    Checks whether an object is a list of (2,) tuples
    """
    if isinstance(L, (list, tuple)):
        for i in L:
            if len(i) != 2:
                raise ValueError("elem i: {} is not of length 2".format(i))
    else:
        raise TypeError("L must be of type [list, tuple]")
    return True


def not_column_float(ser):
    return ser.dtype not in [np.float64, np.float32, np.float, np.float16, float]


def is_column_float(ser):
    return ser.dtype in [np.float64, np.float32, np.float, np.float16, float]


def is_column_object(ser):
    return ser.dtype in [object, pd.CategoricalDtype]


def convert_boolean(df, col, name_map):
    df[col] = df[col].astype(np.bool)
    name_map[col] = "is_"+col


def convert_category(df, col, uniques):
    c_cat = CategoricalDtype(np.sort(uniques), ordered=True)
    df[col] = df[col].astype(c_cat)


def attach_name(*pds):
    return list(it.chain.from_iterable([pd.name_ + "__" + pd.df_.columns for pd in pds]))


def is_boolean_series(bool_s):
    if not isinstance(bool_s, pd.Series):
        raise TypeError("bool_s must be of type [pd.Series], not {}".format(type(bool_s)))
    if not bool_s.dtype in [bool, np.bool]:
        raise TypeError("bool_s must contain booleans, not type '{}'".format(bool_s.dtype))


def check_list_type(L, t):
    for i, l in enumerate(L):
        if not isinstance(l, t):
            raise TypeError("type '{}' not found in list at index [{}]".format(t, i))
    return True


def instance_check(a, i):
    if not isinstance(a, i):
        raise TypeError("object '{}' does not belong to type {}".format(a, i))


def chain_intersection(*cgroup):
    """
    Given a group of pandas.Index, perform intersection on A & B & C & .. & K
    """
    mchain = iter(cgroup)
    res = mchain.__next__()
    for m in mchain:
        res = res.intersection(m)
    return res


def chain_union(*cgroup):
    """
    Given a group of pandas.Index, perform union on A | B | C | .. | K
    """
    mchain = iter(cgroup)
    res = mchain.__next__()
    for m in mchain:
        res = res.union(m)
    return res


def remove_multi_index(df):
    if isinstance(df.columns, pd.MultiIndex):
        indices = [n if n is not None else ("Index%d" % i) for i, n in enumerate(df.columns.names)]
        df.columns = pd.Index(["__".join(col) for col in df.columns], name="__".join(indices))


def remove_string_spaces(df):
    for c in df.columns[df.dtypes.eq(object)]:
        df[c] = df[c].str.strip()
    # if we have an obj index, strip this
    if df.index.dtype == object and (not isinstance(df.index, pd.MultiIndex)):
        df.index = df.index.str.strip()


def calc_mem(df):
    return df.memory_usage().sum()/1000000
