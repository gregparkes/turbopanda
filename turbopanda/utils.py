#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides a host of utility functions."""

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
from typing import Any, Dict, Tuple, TypeVar, Union, List, Optional, Callable, Iterable

# imports
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.stats import norm

from .custypes import ArrayLike, SetLike

# defined custom custypes.py
AsPandas = Union[pd.Series, pd.DataFrame]

__all__ = ("fself", "is_twotuple", "instance_check", "dictzip", "dictmap", "t_numpy",
           "boolean_series_check", "check_list_type", "not_column_float",
           "is_column_float", "is_column_object", "is_column_int",
           "calc_mem", "remove_string_spaces", "nearest_factors", "is_missing",
           "split_file_directory", "c_float", "c_int", "intcat",
           "is_unique_id", "string_replace", "object_to_categorical",
           "is_n_value_column", "boolean_to_integer", "integer_to_boolean",
           "join", "belongs", "is_possible_category",
           "standardize", "dict_to_tuple", "set_like", "union", "difference",
           "intersect", "interacting_set", "is_column_string", "remove_na",
           "common_substring_match", "pairwise")


def c_float() -> Tuple[TypeVar, ...]:
    """Returns accepted float custypes.py."""
    return np.float64, np.float32, np.float16, np.float, float


def c_int() -> Tuple[TypeVar, ...]:
    """Returns accepted integer custypes.py."""
    return (np.int64, np.int32, np.int16, np.int8,
            np.int, np.uint, np.uint8, np.uint16,
            np.uint16, np.uint32, int)


def t_numpy() -> Tuple[TypeVar, ...]:
    """Returns the supported custypes.py from NumPy."""
    return (
        np.int, np.bool, np.float, np.float64, np.float32, np.float16, np.int64,
        np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64
    )


def intcat() -> Tuple[TypeVar, ...]:
    """Returns accepted category custypes.py."""
    return np.uint8, np.uint16, object


def fself(x: Any):
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
    a : list/tuple
        A sequence of some kind
    b : list/tuple
        A sequence of some kind

    Returns
    -------
    d : dict
        Mapped dictionary
    """
    return dict(it.zip_longest(a, b))


def dictmap(a: Iterable, b: Callable) -> Dict:
    """Map together a with the result of function b to make {a: b}.

    b must return something that can be placed in d.

    Parameters
    ----------
    a : list/tuple
        A sequence of some kind
    b : function
        A function b(y) returning something to be placed in d

    Returns
    -------
    d : dict
        Mapped dictionary
    """
    return dict(it.zip_longest(a, map(b, a)))


def list_dir(obj: Any) -> List:
    """Lists all public functions, classes within a list directory."""
    return [a for a in dir(obj) if not a.startswith("__") and not a.startswith("_")]


def is_possible_category(ser: pd.Series) -> bool:
    """Checks whether the data type in Series is categorizable."""
    return ser.dtype in intcat()


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
    return ser.dtype in (object, pd.CategoricalDtype)


def is_missing(ser: pd.Series) -> bool:
    """Determine whether any missing values are present."""
    return ser.count() < ser.shape[0]


def is_n_value_column(ser: pd.Series, n: int = 1) -> bool:
    """Determine whether the number of unique values equals some value n."""
    return ser.nunique() == n


def is_unique_id(ser: pd.Series) -> bool:
    """Determine whether ser is unique."""
    return ser.is_unique if is_column_int(ser) else False


def split_file_directory(filename: str):
    """Breaks down the filename pathway into constitute parts.

    e.g a string "path/to/some_data/datafile.csv" returns:
        ("path/to/some_data", "datafile", "csv")

    Parameters
    --------
    filename : str
        The filename full string

    Returns
    -------
    directory : str
        The directory linking to the file (no `/` at the end)
    jname : str
        The name of the file, (no `.` or extension)
    ext : str
        Extension type without `.`, always returned as lowercase.
        e.g Does not return {'CSV', 'XLSX'} but {'csv', 'xlsx'}
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
    return directory, jname, ext.lower()


def is_twotuple(t: Tuple[Any, Any]) -> bool:
    """Checks whether an object is a list of (2,) tuples."""
    if isinstance(t, (list, tuple)):
        for i in t:
            if len(i) != 2:
                raise ValueError("elem i: {} is not of length 2".format(i))
    else:
        raise TypeError("L must be of type [list, tuple]")
    return True


def string_replace(strings: Union[pd.Series, pd.Index], operations: Tuple[str, str]) -> pd.Series:
    """ Performs all replace operations on the string inplace """
    for op in operations:
        strings = strings.str.replace(*op)
    return strings


def integer_to_boolean(ser: pd.Series) -> pd.Series:
    """ Convert an integer series into boolean if possible """
    return ser.astype(np.bool) if (is_column_int(ser) and is_n_value_column(ser, 2)) else ser


def object_to_categorical(ser: pd.Series, order: Optional[Tuple] = None, thresh: int = 30) -> pd.Series:
    """Convert ser to be of type 'category' if possible."""
    # get uniques if possible
    if 1 < ser.nunique() < thresh:
        if order is None:
            return ser.astype(CategoricalDtype(ser.dropna().unique(), ordered=False))
        else:
            return ser.astype(CategoricalDtype(order, ordered=True))
    else:
        return ser


def boolean_to_integer(ser: pd.Series) -> pd.Series:
    """ Convert a boolean series into an integer if possible """
    return ser.astype(np.uint8) if (ser.dtype == np.bool) else ser


def boolean_series_check(ser: pd.Series):
    """Check whether ser is full of booleans or not."""
    if not isinstance(ser, pd.Series):
        raise TypeError("bool_s must be of type [pd.Series], not {}".format(type(ser)))
    if ser.dtype not in (bool, np.bool):
        raise TypeError("bool_s must contain booleans, not type '{}'".format(ser.dtype))


def check_list_type(elems: Tuple, t: TypeVar) -> bool:
    """Checks the type of every element in the list."""
    for i, elem in enumerate(elems):
        if not isinstance(elem, t):
            raise TypeError("type '{}' not found in list at index [{}]".format(t, i))
    return True


def belongs(elem: Any, home: Union[List[Any], Tuple[Any, ...]]):
    """Check whether elem belongs in a list."""
    if elem not in home:
        raise ValueError("element {} is not found in list: {}".format(elem, home))


def instance_check(a: object, i: TypeVar):
    """Check that a is an instance of type i."""
    if isinstance(i, str):
        if not hasattr(a, i):
            raise AttributeError("object '{}' does not have attribute '{}'".format(a, i))
    elif not isinstance(a, i):
        raise TypeError("object '{}' does not belong to type {}".format(a, i))
    elif isinstance(i, (list, tuple)):
        if None in i and a is not None:
            raise TypeError("object '{}' is not of type None".format(a))


def join(*pipes: Optional[Iterable[Any]]) -> List:
    """Perform it.chain.from_iterable on iterables."""
    # filter out None elements.
    pipes = list(filter(None.__ne__, pipes))
    # use itertools to chain together elements.
    return list(it.chain.from_iterable(pipes))


def pairwise(f, x):
    """Conduct a pairwise operation on a list of elements, receiving them in pairs.

    e.g for list x = [1, 2, 3] we conduct:
        for i in range(3):
            for j in range(i, 3):
                operation[i, j]

    Parameters
    ----------
    f : function
        Receives two arguments and returns something
    x : list-like
        A list of strings, parameters etc, to pass to f

    Returns
    -------
    y : list-like
        A list of the return elements from f(x)
    """
    y = []
    pairs = it.combinations(x, 2)
    for p1, p2 in pairs:
        y.append(f(p1, p2))
    return join(y)


def set_like(x: SetLike = None) -> pd.Index:
    """
    Convert x to something unique, set-like.

    Parameters
    ----------
    x : str, list, tuple, pd.Series, set, pd.Index, optional
        A variable that can be made set-like.
        strings are wrapped.

    Returns
    -------
    y : pd.Index
        Set-like result.
    """
    if x is None:
        return pd.Index([])
    if isinstance(x, str):
        return pd.Index([x])
    if isinstance(x, (list, tuple)):
        return pd.Index(set(x))
    elif isinstance(x, (pd.Series, pd.Index)):
        return pd.Index(x.dropna().unique(), name=x.name)
    elif isinstance(x, set):
        return pd.Index(x)
    else:
        raise TypeError(
            "x must be in {}, not of type {}".format(
                ['None', 'str', 'list', 'tuple', 'pd.Series', 'pd.Index', 'set'], type(x)))


def union(*args: SetLike) -> pd.Index:
    """Performs set union all passed arguments, whatever type they are.

    Parameters
    ----------
    args : str, list, tuple, pd.Series, set, pd.Index
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


def intersect(*args: SetLike) -> pd.Index:
    """Performs set intersect all passed arguments, whatever type they are.

    Parameters
    ----------
    args : str, list, tuple, pd.Series, set, pd.Index
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


def difference(a: SetLike, b: SetLike) -> pd.Index:
    """
    Performs set symmetric difference on a and b, whatever type they are.

    Parameters
    ----------
    a : str, list, tuple, pd.Series, set, pd.Index
        List-like a
    b : str, list, tuple, pd.Series, set, pd.Index
        List-like b

    Returns
    -------
    c : pd.Index
        Symmetric difference between a & b
    """
    return set_like(a).symmetric_difference(set_like(b))


def interacting_set(sets):
    """
    Given a list of pd.Index, calculates whether any of the values are shared
    between any of the indexes.
    """
    union_l = []
    # generate a list of potential interactions.
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            interact = intersect(sets[i], sets[j])
            union_l.append(interact)

    return union(*union_l)


def is_column_string(ser: Union[pd.Series, pd.Index]) -> bool:
    """Determines whether the column can operate on strings."""
    try:
        ser.dropna().str.contains("TestString_Hello")
        return True
    except AttributeError:
        return False


def remove_string_spaces(df: pd.DataFrame):
    """Performs strip on df.column names."""
    for c in df.columns[df.dtypes.eq(object)]:
        df[c] = df[c].str.strip()
    # if we have an obj index, strip this
    if df.index.dtype == object and (not isinstance(df.index, pd.MultiIndex)):
        df.index = df.index.str.strip()


def calc_mem(df: AsPandas) -> float:
    """Calculate the memory usage in megabytes."""
    return (df.memory_usage().sum() / 1000000.) if (df.ndim > 1) else (df.memory_usage() / 1000000.)


def _factor(n: int):
    """
    Collect a list of factors given an integer, excluding 1 and n
    """
    if not isinstance(n, (int, np.int, np.int64)):
        raise TypeError("'n' must be an integer")

    # noinspection PyRedundantParentheses
    def prime_powers(_n):
        """c goes through 2, 3, 5 then the infinite (6n+1, 6n+5) series."""
        for c in it.accumulate(it.chain([2, 1, 2], it.cycle([2, 4]))):
            if c * c > _n:
                break
            if _n % c:
                continue
            d, p = (), c
            while not _n % c:
                _n, p, d = _n // c, p * c, d + (p,)
            yield (d)
        if _n > 1:
            yield ((_n,))

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
        rng = np.arange(n, n + search_range, 1, dtype=np.int64)
        # calculate new scores - using function
        nscores = np.asarray([f_(i) for i in rng])
        # calculate distance
        dist = np.abs(nscores[:, 0] - nscores[:, 1])
        # weight our distances by a normal distribution -
        # we don't want to generate too many plots!
        w_dist = dist * (1. - norm.pdf(rng, n, w_var))
        # calculate new N
        return tuple(nscores[w_dist.argmin()])
    else:
        return a, b


def standardize(x: ArrayLike) -> ArrayLike:
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


def _remove_na_single(x, axis='rows'):
    """Remove NaN in a single array.
    This is an internal Pingouin function.
    """
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
    else:
        # 2D arrays
        ax = 1 if axis == 'rows' else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
    # Check if missing values are present
    if ~x_mask.all():
        ax = 0 if axis == 'rows' else 1
        ax = 0 if x.ndim == 1 else ax
        x = x.compress(x_mask, axis=ax)
    return x


def remove_na(x: np.ndarray, y: np.ndarray = None, paired=False, axis='rows'):
    """Remove missing values along a given axis in one or more (paired) numpy
    arrays.

    Adapted from the `pingouin` library, made by Raphael Vallat.

    .. [1] https://github.com/raphaelvallat/pingouin/blob/master/pingouin/correlation.py

    Parameters
    ----------
    x, y : np.ndarray, array like
        Data. ``x`` and ``y`` must have the same number of dimensions.
        ``y`` can be None to only remove missing values in ``x``.
    paired : bool
        Indicates if the measurements are paired or not.
    axis : str
        Axis or axes along which missing values are removed.
        Can be 'rows' or 'columns'. This has no effect if ``x`` and ``y`` are
        one-dimensional arrays.

    Returns
    -------
    x, y : np.ndarray
        Data without missing values
    """
    # Safety checks
    x = np.asarray(x)
    assert x.size > 1, 'x must have more than one element.'
    assert axis in ['rows', 'columns'], 'axis must be rows or columns.'

    if y is None:
        return _remove_na_single(x, axis=axis)
    elif isinstance(y, (int, float, str)):
        return _remove_na_single(x, axis=axis), y
    elif isinstance(y, (list, np.ndarray)):
        y = np.asarray(y)
        # Make sure that we just pass-through if y have only 1 element
        if y.size == 1:
            return _remove_na_single(x, axis=axis), y
        if x.ndim != y.ndim or paired is False:
            # x and y do not have the same dimension
            x_no_nan = _remove_na_single(x, axis=axis)
            y_no_nan = _remove_na_single(y, axis=axis)
            return x_no_nan, y_no_nan

    # At this point, we assume that x and y are paired and have same dimensions
    if x.ndim == 1:
        # 1D arrays
        x_mask = ~np.isnan(x)
        y_mask = ~np.isnan(y)
    else:
        # 2D arrays
        ax = 1 if axis == 'rows' else 0
        x_mask = ~np.any(np.isnan(x), axis=ax)
        y_mask = ~np.any(np.isnan(y), axis=ax)

    # Check if missing values are present
    if ~x_mask.all() or ~y_mask.all():
        ax = 0 if axis == 'rows' else 1
        ax = 0 if x.ndim == 1 else ax
        both = np.logical_and(x_mask, y_mask)
        x = x.compress(both, axis=ax)
        y = y.compress(both, axis=ax)
    return x, y


def common_substring_match(a, b):
    """Given two strings, find the longest common substring.

     Also known as the Longest Common Substring problem."""
    from difflib import SequenceMatcher
    match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    # return the longest substring
    if match.size != 0:
        return a[match.a:match.a + match.size]
    else:
        raise ValueError("in 'common_substring_match', no match was found")
