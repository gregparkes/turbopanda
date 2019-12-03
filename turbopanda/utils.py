#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:13:38 2019

@author: gparkes
"""

import numpy as np
import pandas as pd
from scipy import stats
import itertools as it
from pandas.api.types import CategoricalDtype


__all__ = ["is_twotuple", "instance_check", "chain_intersection", "chain_union",
           "is_boolean_series", "attach_name", "check_list_type", "not_column_float",
           "is_column_float", "is_column_object", "convert_category", "convert_boolean",
           "calc_mem", "remove_multi_index", "remove_string_spaces", "check_pipe_attr",
           "nearest_square_factors"]


def is_twotuple(t):
    """
    Checks whether an object is a list of (2,) tuples
    """
    if isinstance(t, (list, tuple)):
        for i in t:
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


def is_missing_values(ser):
    return ser.count() < ser.shape[0]


def convert_boolean(df, col, name_map):
    df[col] = df[col].astype(np.bool)
    name_map[col] = "is_" + col


def convert_category(df, col, uniques):
    c_cat = CategoricalDtype(np.sort(uniques), ordered=True)
    df[col] = df[col].astype(c_cat)


def attach_name(*pds):
    return list(it.chain.from_iterable([df.name_ + "__" + df.df_.columns for df in pds]))


def is_boolean_series(bool_s):
    if not isinstance(bool_s, pd.Series):
        raise TypeError("bool_s must be of type [pd.Series], not {}".format(type(bool_s)))
    if bool_s.dtype not in [bool, np.bool]:
        raise TypeError("bool_s must contain booleans, not type '{}'".format(bool_s.dtype))


def check_list_type(l, t):
    for i, elem in enumerate(l):
        if not isinstance(elem, t):
            raise TypeError("type '{}' not found in list at index [{}]".format(t, i))
    return True


def instance_check(a, i):
    if not isinstance(a, i):
        raise TypeError("object '{}' does not belong to type {}".format(a, i))


def check_pipe_attr(obj, l):
    for i in l:
        if len(i) != 3:
            raise ValueError("pipe element {} needs to be of length 3".format(i))
        if not hasattr(obj, i[0]):
            raise ValueError("elem {} not found as attribute in obj {}".format(i[0], obj))
        if not isinstance(i[1], (list, tuple)):
            raise TypeError("elem {} not belong to type [list, tuple], but {}".format(i[1], type(i[1])))
        if not isinstance(i[2], dict):
            raise TypeError("elem {} not belong to type [dict], but {}".format(i[2], type(i[2])))


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
    return (df.memory_usage().sum() / 1000000.) if (df.ndim > 1) else (df.memory_usage() / 1000000.)


def factor(n):
    """
    Collect a list of factors given an integer, excluding 1 and n
    """
    if not isinstance(n, (int, np.int, np.int64)):
        raise TypeError("'n' must be an integer")

    def prime_powers(n):
        # c goes through 2, 3, 5 then the infinite (6n+1, 6n+5) series
        for c in it.accumulate(it.chain([2, 1, 2], it.cycle([2, 4]))):
            if c * c > n: break
            if n % c: continue
            d, p = (), c
            while not n % c:
                n, p, d = n // c, p * c, d + (p,)
            yield (d)
        if n > 1: yield ((n,))

    r = [1]
    for e in prime_powers(n):
        r += [a * b for a in r for b in e]
    return r


def square_factors(n):
    """
    Given n size, calculate the 'most square' factors of that integer.

    Parameters
    -------
    n : int
        An *even* integer that is factorizable.

    Returns
    -------
    F_t : tuple (2,)
        Two integers representing the most 'square' factors.
    """
    if not isinstance(n, (int, np.int, np.int64)):
        raise TypeError("'n' must of type [int, np.int, np.int64]")
    arr = np.sort(np.asarray(factor(n)))
    return arr[arr.shape[0] // 2], arr[-1] // arr[arr.shape[0] // 2]


def nearest_square_factors(n, cutoff=6, search_range=5, W_var=1.5):
    """
    Given n size that may not be even, return the 'most square' factors
    of that integer. Uses square_factors and searches linearly around
    options.

    Parameters
    -------
    n : int
        An integer.
    cutoff : int
        The distance between factors whereby any higher requires a search
    search_range : int
        The number of characters forward to search in
    W_var : float
        The variance applied to the normal distribution weighting

    Returns
    -------
    F_t : tuple (2,)
        Two integers representing the most 'square' factors.
    """
    a, b = square_factors(n)
    # if our 'best' factors don't cut it...
    if abs(a - b) > cutoff:
        # create Range
        R = np.arange(n, n + search_range, 1, dtype=np.int64)
        # calculate new scores
        nscores = np.asarray([square_factors(i) for i in R])
        # calculate distance
        dist = np.abs(nscores[:, 0] - nscores[:, 1])
        # weight our distances by a normal distribution -
        # we don't want to generate too many plots!
        w_dist = dist * (1. - stats.norm.pdf(R, n, W_var))
        # calculate new N
        return tuple(nscores[w_dist.argmin()])
    else:
        return a, b
