#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:13:38 2019

@author: gparkes
"""

import os
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import warnings

from pandas.api.types import CategoricalDtype
from scipy.stats import norm

__all__ = ["is_twotuple", "instance_check", "chain_intersection", "chain_union",
           "boolean_series_check", "check_list_type", "not_column_float",
           "is_column_float", "is_column_object", "is_column_int",
           "calc_mem", "remove_string_spaces", "nearest_factors", "is_missing_values",
           "is_unique_id", "is_potential_id",
           "is_potential_stacker", "nunique", "object_to_categorical",
           "is_n_value_column", "boolean_to_integer", "integer_to_boolean",
           "is_metapanda_pipe", "join", "belongs", "is_possible_category"]


def _cfloat():
    return [np.float64, np.float32, np.float16, np.float, float]


def _cint():
    return [np.int64, np.int32, np.int16, np.int8, np.int, np.uint, np.uint8, np.uint16, np.uint16, np.uint32, int]


def _intcat():
    return [np.uint8, np.uint16]


def is_possible_category(ser):
    return ser.dtype in ([object] + _intcat())


def not_column_float(ser):
    return ser.dtype not in _cfloat()


def is_column_float(ser):
    return ser.dtype in _cfloat()


def is_column_int(ser):
    return ser.dtype in _cint()


def is_column_object(ser):
    return ser.dtype in [object, pd.CategoricalDtype]


def is_missing_values(ser):
    return ser.count() < ser.shape[0]


def is_n_value_column(ser, n=1):
    return nunique(ser) == n


def is_unique_id(ser):
    # a definite algorithm for determining a unique column IDm
    return ser.is_unique if is_column_int(ser) else False


def is_potential_id(ser, thresh=0.5):
    return (ser.unique().shape[0] / ser.shape[0]) > thresh if is_column_int(ser) else False


def is_potential_stacker(ser, regex=";|\t|,|", thresh=0.1):
    return ser.dropna().str.contains(regex).sum() > thresh if (ser.dtype == object) else False


def is_metapanda_pipe(p):
    for pipe in p:
        if len(pipe) != 3:
            raise ValueError("pipe of length 3 is of length {}".format(len(pipe)))
        # element 1: string
        instance_check(pipe[0], str)
        instance_check(pipe[1], (list, tuple))
        instance_check(pipe[2], dict)
    return True


def nunique(ser):
    return ser.nunique() if not_column_float(ser) else -1


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


def integer_to_boolean(ser):
    """ Convert an integer series into boolean if possible """
    return ser.astype(np.bool) if (is_column_int(ser) and is_n_value_column(ser, 2)) else ser


def object_to_categorical(ser, thresh=30):
    # get uniques if possible
    if 1 < nunique(ser) < thresh:
        c_cat = CategoricalDtype(np.sort(ser.dropna().unique()), ordered=True)
        return ser.astype(c_cat)
    else:
        return ser


def boolean_to_integer(ser):
    """ Convert a boolean series into an integer if possible """
    return ser.astype(np.uint8) if (ser.dtype == np.bool) else ser


def boolean_series_check(ser):
    if not isinstance(ser, pd.Series):
        raise TypeError("bool_s must be of type [pd.Series], not {}".format(type(ser)))
    if ser.dtype not in [bool, np.bool]:
        raise TypeError("bool_s must contain booleans, not type '{}'".format(ser.dtype))


def check_list_type(l, t):
    for i, elem in enumerate(l):
        if not isinstance(elem, t):
            raise TypeError("type '{}' not found in list at index [{}]".format(t, i))
    return True


def belongs(elem, l):
    if elem not in l:
        raise ValueError("element {} is not found in list: {}".format(elem,l))


def instance_check(a, i):
    if not isinstance(a, i):
        raise TypeError("object '{}' does not belong to type {}".format(a, i))


def join(*pipes):
    return list(it.chain.from_iterable(pipes))


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


def remove_string_spaces(df):
    for c in df.columns[df.dtypes.eq(object)]:
        df[c] = df[c].str.strip()
    # if we have an obj index, strip this
    if df.index.dtype == object and (not isinstance(df.index, pd.MultiIndex)):
        df.index = df.index.str.strip()


def calc_mem(df):
    return (df.memory_usage().sum() / 1000000.) if (df.ndim > 1) else (df.memory_usage() / 1000000.)


def _factor(n):
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


def _square_factors(n):
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
    arr = np.sort(np.asarray(_factor(n)))
    return arr[arr.shape[0] // 2], arr[-1] // arr[arr.shape[0] // 2]


def _diag_factors(n):
    """
    Given n size, calculate the 'most off-edge' factors of that integer.

    Parameters
    -------
    n : int
        An *even* integer that is factorizable.

    Returns
    -------
    F_t : tuple (2,)
        Two integers representing the most 'off-edge' factors.
    """
    if not isinstance(n, (int, np.int, np.int64)):
        raise TypeError("'n' must of type [int, np.int, np.int64]")
    arr = np.sort(np.asarray(_factor(n)))
    return arr[arr.shape[0] // 4], arr[-1] // arr[arr.shape[0] // 4]


def nearest_factors(n, ftype="square", cutoff=6, search_range=5, W_var=1.5):
    """
    Given n size that may not be even, return the 'most square' factors
    of that integer. Uses square_factors and searches linearly around
    options.

    Parameters
    -------
    n : int
        An integer.
    ftype : str
        ['diag' or 'square'], by default uses square factors.
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
    if ftype=="square":
        f_ = _square_factors
    elif ftype=="diag":
        f_ = _diag_factors
    else:
        raise ValueError("ftype must be [diag, square]")

    a, b = f_(n)

    # if our 'best' factors don't cut it...
    if abs(a - b) > cutoff:
        # create Range
        R = np.arange(n, n + search_range, 1, dtype=np.int64)
        # calculate new scores - using function
        nscores = np.asarray([f_(i) for i in R])
        # calculate distance
        dist = np.abs(nscores[:, 0] - nscores[:, 1])
        # weight our distances by a normal distribution -
        # we don't want to generate too many plots!
        w_dist = dist * (1. - norm.pdf(R, n, W_var))
        # calculate new N
        return tuple(nscores[w_dist.argmin()])
    else:
        return a, b


def save_figure(fig_obj,
                plot_type,
                name="example1",
                save_types=["png", "pdf"],
                fp="./",
                dpi=360,
                savemode="first"):
    """
    Given a matplotlib.Figure object, save appropriate numbers of Figures to the respective
    folders.

    Parameters
    -------
    fig : matplotlib.Figure
        The figure object to save.
    plot_type : str
        The type of plot this is, accepted inputs are:
        ["scatter", "kde", "heatmap", "cluster", "bar", "hist", "kde", "quiver",
        "box", "line", "venn", "multi", "pie"]
    name : str (optional)
        The name of the file, this may be added to based on the other parameters
    save_types : list (optional)
        Contains every unique save type to use e.g ["png", "pdf", "svg"]..
    fp : str (optional)
        The file path to the root directory of saving images
    dpi : int
        The resolution in dots per inch; set to high if you want a good image
    savemode : str
        ['first', 'update']: if first, only saves if file isn't present, if update,
        overrides saved figure

    Returns
    -------
    success : bool
        Whether it was successful or not
    """
    instance_check(fig_obj, plt.Figure)
    accepted_types = [
        "scatter", "kde", "heatmap", "cluster", "bar", "hist", "kde", "quiver",
        "box", "line", "venn", "multi", "pie"
    ]
    file_types_supported = ["png", "pdf", "svg", "eps", "ps"]
    accepted_savemodes = ['first', 'update']

    if plot_type not in accepted_types:
        raise TypeError("plot_type: [%s] not found in accepted types!" % plot_type)

    for st in save_types:
        if st not in file_types_supported:
            TypeError("save_type: [%s] not supported" % st)
    if savemode not in accepted_savemodes:
        raise ValueError("savemode: '{}' not found in {}".format(savemode, accepted_savemodes))

    # correct to ensure filepath has / at end
    if not fp.endswith("/"):
        fp += "/"

    # check whether the filepath exists
    if os.path.exists(fp):
        for t in save_types:
            # if the directory does not exist, create it!
            if not os.path.isdir(fp + "_" + t):
                os.mkdir(fp + "_" + t)
            # check if the figures themselves already exist.
            filename = "{}_{}/{}_{}.{}".format(fp, t, plot_type, name, t)
            if os.path.isfile(filename):
                warnings.warn("Figure: '{}' already exists: Using savemode: {}".format(filename, savemode), UserWarning)
                if savemode == 'update':
                    fig_obj.savefig(filename, format=t, bbox_inches='tight', dpi=dpi)
            else:
                # make the file
                fig_obj.savefig(filename, format=t, bbox_inches="tight", dpi=dpi)
    else:
        raise IOError("filepath: [%s] does not exist." % fp)
    return True
