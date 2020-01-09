#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import numpy as np
import pandas as pd
import itertools as it
from pandas.api.types import CategoricalDtype
from scipy.stats import norm

from typing import Any, Dict, Tuple, Set, Iterable, TypeVar, Sequence

__all__ = ("fself", "is_twotuple", "instance_check", "dictzip",
           "boolean_series_check", "check_list_type", "not_column_float",
           "is_column_float", "is_column_object", "is_column_int",
           "calc_mem", "remove_string_spaces", "nearest_factors", "is_missing_values",
           "split_file_directory", "c_float", "c_int", "intcat",
           "is_unique_id", "is_potential_id", "string_replace",
           "is_potential_stacker", "nunique", "object_to_categorical",
           "is_n_value_column", "boolean_to_integer", "integer_to_boolean",
           "is_metapanda_pipe", "join", "belongs", "is_possible_category",
           "standardize", "dict_to_tuple", "set_like", "union", "difference", "intersect")


def c_float() -> Set:
    """Returns accepted float types."""
    return {np.float64, np.float32, np.float16, np.float, float}


def c_int() -> Set:
    """Returns accepted integer types."""
    return {np.int64, np.int32, np.int16, np.int8, np.int, np.uint, np.uint8, np.uint16, np.uint16, np.uint32, int}


def intcat() -> Set:
    """Returns accepted category types."""
    return {np.uint8, np.uint16}


def fself(x):
    """Self lambda function."""
    return x


def dict_to_tuple(d: Dict) -> Tuple:
    """Converts a dictionary to a 2-tuple."""
    return tuple((a, b) for a, b in d.items())


def dictzip(a: Iterable, b: Iterable) -> Dict:
    """Map together a, b to make {a: b}.

    a and b must be the same length.

    Parameters
    ----------
    a : iterable
        A sequence of some kind
    b : iterable
        A sequence of some kind

    Returns
    -------
    d : dict
        Mapped dictionary
    """
    return dict(it.zip_longest(a, b))


def list_dir(obj: Any) -> Tuple:
    """Lists all public functions, classes within a list directory."""
    return (a for a in dir(obj) if not a.startswith("__") and not a.startswith("_"))


def is_possible_category(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is categorizable."""
    return ser.dtype in ({object} | intcat())


def not_column_float(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is not a float column."""
    return ser.dtype not in c_float()


def is_column_float(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is a float column."""
    return ser.dtype in c_float()


def is_column_int(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is a integer column."""
    return ser.dtype in c_int()


def is_column_object(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is a object column."""
    return ser.dtype in {object, pd.CategoricalDtype}


def is_missing_values(ser: pd.Series) -> bool:
    return ser.count() < ser.shape[0]


def is_n_value_column(ser: pd.Series, n: int = 1):
    return nunique(ser) == n


def is_unique_id(ser: pd.Series) -> bool:
    # a definite algorithm for determining a unique column ID
    return ser.is_unique if is_column_int(ser) else False


def is_potential_id(ser: pd.Series, thresh: float = 0.5) -> bool:
    return (ser.unique().shape[0] / ser.shape[0]) > thresh if is_column_int(ser) else False


def is_potential_stacker(ser: pd.Series, regex: str = ";|\t|,|", thresh: float = 0.1) -> bool:
    return ser.dropna().str.contains(regex).sum() > thresh if (ser.dtype == object) else False


def split_file_directory(filename: str):
    """Breaks down the filename pathway into constitute parts.

    Parameters
    --------
    filename : str
        The filename full string

    Returns
    -------
    directory : str
        The directory linking to the file
    jname : str
        The name of the file (without extension)
    ext : str
        Extension type
    """
    fs = filename.rsplit("/", 1)
    if len(fs) == 0:
        raise ValueError("filename '{}' not recognized".format(filename))
    elif len(fs) == 1:
        directory = "."
        fname = fs[0]
    else:
        directory, fname = fs
    # just the name without the extension
    jname, ext = fname.split(".", 1)
    return directory, jname, ext


def is_metapanda_pipe(p):
    for pipe in p:
        if len(pipe) != 3:
            raise ValueError("pipe of length 3 is of length {}".format(len(pipe)))
        # element 1: string
        instance_check(pipe[0], str)
        instance_check(pipe[1], (list, tuple))
        instance_check(pipe[2], dict)
    return True


def nunique(ser: pd.Series) -> bool:
    return ser.nunique() if not_column_float(ser) else -1


def is_twotuple(t: Tuple) -> bool:
    """Checks whether an object is a list of (2,) tuples."""
    if isinstance(t, (list, tuple)):
        for i in t:
            if len(i) != 2:
                raise ValueError("elem i: {} is not of length 2".format(i))
    else:
        raise TypeError("L must be of type [list, tuple]")
    return True


def string_replace(l, operations: Tuple):
    """ Performs all replace operations on the string inplace """
    for op in operations:
        l = l.str.replace(*op)
    return l


def integer_to_boolean(ser: pd.Series) -> pd.Series:
    """ Convert an integer series into boolean if possible """
    return ser.astype(np.bool) if (is_column_int(ser) and is_n_value_column(ser, 2)) else ser


def object_to_categorical(ser: pd.Series, order=None, thresh: int = 30):
    # get uniques if possible
    if 1 < nunique(ser) < thresh:
        if order is None:
            return ser.astype(CategoricalDtype(np.sort(ser.dropna().unique()), ordered=True))
        else:
            return ser.astype(CategoricalDtype(order, ordered=True))
    else:
        return ser


def boolean_to_integer(ser: pd.Series) -> pd.Series:
    """ Convert a boolean series into an integer if possible """
    return ser.astype(np.uint8) if (ser.dtype == np.bool) else ser


def boolean_series_check(ser: pd.Series):
    if not isinstance(ser, pd.Series):
        raise TypeError("bool_s must be of type [pd.Series], not {}".format(type(ser)))
    if ser.dtype not in [bool, np.bool]:
        raise TypeError("bool_s must contain booleans, not type '{}'".format(ser.dtype))


def check_list_type(l: Iterable, t: TypeVar) -> bool:
    for i, elem in enumerate(l):
        if not isinstance(elem, t):
            raise TypeError("type '{}' not found in list at index [{}]".format(t, i))
    return True


def belongs(elem: Any, l: Iterable):
    if elem not in l:
        raise ValueError("element {} is not found in list: {}".format(elem, l))


def instance_check(a: object, i: TypeVar):
    if not isinstance(a, i):
        raise TypeError("object '{}' does not belong to type {}".format(a, i))
    elif isinstance(i, (list, tuple)):
        if None in i and a is not None:
            raise TypeError("object '{}' is not of type None".format(a))


def join(*pipes):
    return list(it.chain.from_iterable(pipes))


def set_like(x: Sequence) -> pd.Index:
    """
    Convert x to something unique, set-like.

    Parameters
    ----------
    x : list, tuple, pd.Series, set, pd.Index
        A list of variables

    Returns
    -------
    y : pd.Index
        Set-like result.
    """
    if isinstance(x, (list, tuple)):
        return pd.Index(set(x))
    elif isinstance(x, (pd.Series, pd.Index)):
        return pd.Index(x.dropna().unique())
    elif isinstance(x, set):
        return pd.Index(x)
    else:
        raise TypeError(
            "x must be in {}, not of type {}".format(['list', 'tuple', 'pd.Series', 'pd.Index', 'set'], type(x)))


def union(*args) -> pd.Index:
    """Performs set union all passed arguments, whatever type they are.

    Parameters
    ----------
    args : list, tuple, pd.Series, set, pd.Index
        k List-like arguments

    Raises
    ------
    ValueError
        There must be at least two arguments

    Returns
    -------
    U : pd.Index
        Union between a | b | ... | k
    """
    if len(args) == 0:
        raise ValueError('no arguments passed')
    elif len(args) == 1:
        return set_like(args[0])
    else:
        a = set_like(args[0])
        for b in args[1:]:
            a |= set_like(b)
        return a


def intersect(*args) -> pd.Index:
    """Performs set intersect all passed arguments, whatever type they are.

    Parameters
    ----------
    args : list, tuple, pd.Series, set, pd.Index
        k List-like arguments

    Returns
    -------
    I : pd.Index
        Intersect between a & b & ... & k
    """
    if len(args) == 0:
        raise ValueError('no arguments passed')
    elif len(args) == 1:
        return set_like(args[0])
    else:
        a = set_like(args[0])
        for b in args[1:]:
            a &= set_like(b)
        return a


def difference(a: Sequence, b: Sequence) -> pd.Index:
    """
    Performs set symmetric difference on a and b, whatever type they are.

    Parameters
    ----------
    a : list, tuple, pd.Series, set, pd.Index
        List-like a
    b : list, tuple, pd.Series, set, pd.Index
        List-like b

    Returns
    -------
    c : pd.Index
        Symmetric difference between a & b
    """
    return set_like(a).symmetric_difference(set_like(b))


def remove_string_spaces(df: pd.DataFrame):
    for c in df.columns[df.dtypes.eq(object)]:
        df[c] = df[c].str.strip()
    # if we have an obj index, strip this
    if df.index.dtype == object and (not isinstance(df.index, pd.MultiIndex)):
        df.index = df.index.str.strip()


def calc_mem(df: pd.DataFrame) -> float:
    return (df.memory_usage().sum() / 1000000.) if (df.ndim > 1) else (df.memory_usage() / 1000000.)


def _factor(n: int):
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


def _square_factors(n: int):
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


def _diag_factors(n: int):
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


def nearest_factors(n: int, shape: str = "square", cutoff: int = 6, search_range: int = 5, w_var: float = 1.5) -> Tuple:
    """Calculate the nearest best factors of a given integer.


    Given n size that may not be even, return the 'most square' factors
    of that integer. Uses square_factors and searches linearly around
    options.

    Parameters
    -------
    n : int
        An integer.
    shape : str
        ['diag' or 'square'], by default uses square factors.
    cutoff : int
        The distance between factors whereby any higher requires a search
    search_range : int
        The number of characters forward to search in
    w_var : float
        The variance applied to the normal distribution weighting

    Returns
    -------
    F_t : tuple (2,)
        Two integers representing the most 'square' factors.
    """
    if shape == "square":
        f_ = _square_factors
    elif shape == "diag":
        f_ = _diag_factors
    else:
        raise ValueError("ftype must be {'diag', 'square'}")

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
        w_dist = dist * (1. - norm.pdf(R, n, w_var))
        # calculate new N
        return tuple(nscores[w_dist.argmin()])
    else:
        return a, b


def standardize(x):
    """
    Performs z-score standardization on vector x.

    Accepts x as [np.ndarray, pd.Series, pd.DataFrame]
    """
    if isinstance(x, pd.Series):
        return (x - x.mean()) / x.std()
    elif isinstance(x, pd.DataFrame):
        return (x - x.mean(axis=0)) / x.std(axis=0)
    elif isinstance(x, np.ndarray):
        return (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    else:
        raise TypeError("x must be of type [pd.Series, pd.DataFrame, np.ndarray]")
