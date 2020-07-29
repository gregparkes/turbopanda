#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the map function, with parallelism and caching enabled as needed."""

import os
from typing import Callable
import itertools as it
from joblib import load, dump, delayed, Parallel, cpu_count
import pandas as pd
import numpy as np

from ._cache import cache
from ._files import insert_suffix as add_suf
from ._error_raise import is_iterable

__all__ = ('zipe', 'umap', 'umapc', 'umapp', 'umapcc', 'umappc', 'umappcc')


def _downcast_to_list(arg):
    if isinstance(arg, np.ndarray):
        return list(arg.flatten())
    elif isinstance(arg, (pd.Series, pd.Index)):
        return arg.values.tolist()
    elif isinstance(arg, tuple):
        return list(arg)
    else:
        return arg


def zipe(*args):
    """An extension to the zip() function.

    This function converts single-element arguments into the longest-length argument. Note that
    we use `it.zip_longest` and hence differing list lengths will introduce None elements.

    Examples
    --------
    This works like a normal zip function when mapping arguments:
    >>> from turbopanda.utils import zipe
    >>> zipe([1, 3, 5], [2, 4, 6])
    >>> [(1, 2), (3, 4), (5, 6)]
    Further to this, single argument lists are converts to the maximum length one by duplication:
    >>> zipe(1, [4, 6, 8])
    >>> [(1, 4), (1, 6), (1, 8)]
    This extends to more than two arguments, to k:
    >>> zipe(3, 5, [1, 3], [2, 4, 6])
    >>> [(3, 5, 1, 2), (3, 5, 3, 4), (3, 5, None, 6)]
    """
    if len(args) == 1:
        return args[0]
    else:
        _maxlen = max(map(len, filter(lambda x: isinstance(x, list), args)))

        def _singleton(arg):
            return [arg] * _maxlen if not isinstance(arg, list) else arg

        return list(it.zip_longest(map(_singleton, map(_downcast_to_list, args))))


def _map_comp(f, *args):
    # check to make sure every argument is an iterable, and make it one if not
    if len(args) == 0:
        return f()
    elif len(args) == 1 and not is_iterable(args[0]):
        raise TypeError("single argument must be an 'iterable' not '{}'".format(type(args[0])))
    else:
        return list(map(f, *args))


def _parallel_list_comprehension(f, *args):
    if len(args) == 0:
        return f()
    else:
        n = len(args[0])
        ncpu = n if n < cpu_count() else (cpu_count() - 1)
        if len(args) == 1:
            um = Parallel(ncpu)(delayed(f)(arg) for arg in args[0])
        else:
            um = Parallel(ncpu)(delayed(f)(*arg) for arg in zip(*args))
        return um


def umap(f: Callable, *args):
    """Performs Map list comprehension.

    Given function f(x) and arguments a, ..., k; map f(a_i, ..., k_i), ..., f(a_z, ..., k_z) .

    Parameters
    ----------
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args), as a, ..., k

    Returns
    -------
    res : list
        The results from f(*args) or from file

    Examples
    --------
    Provides a clean way to do a list comprehension:
    >>> import turbopanda as turb
    >>> turb.utils.umap(lambda x: x**2, [2, 4, 6])
    >>> [4, 16, 36]
    Like the normal mapping, multiple lists map to multiple parameters passed to the function:
    >>> turb.utils.umap(lambda x, y: x + y, [2, 4, 6], [1, 2, 4])
    >>> [3, 6, 10]
    """
    # check to make sure every argument is an iterable, and make it one if not
    return _map_comp(f, *args)


def umapc(fn: str, f: Callable, *args):
    """Performs Map comprehension with final state Cache.

    That is to say that the first time this runs, function f(*args) is called, storing a cache file.
        The second time and onwards, the resulting cached file is read and no execution takes place.

    Parameters
    ----------
    fn : str
        The path and filename.
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)

    Returns
    -------
    res : Any
        The results from f(*args) or from file

    Examples
    --------
    See `turb.utils.umap` for examples.
    """
    if os.path.isfile(fn):
        # use joblib.load to read in the data
        print("loading file '%s'" % fn)
        return load(fn)
    else:
        # perform list comprehension
        um = _map_comp(f, *args)
        dump(um, fn)
        return um


def umapp(f: Callable, *args):
    """Performs Map comprehension with Parallelism.

    This assumes each iteration is independent from each other in the list comprehension.

    Parameters
    ----------
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)

    Returns
    -------
    res : Any
        The results from f(*args) or from file

    Examples
    --------
    See `turb.utils.umap` for examples.
    """
    return _parallel_list_comprehension(f, *args)


def umappc(fn: str, f: Callable, *args):
    """Performs Map comprehension with Parallelism and Caching.

    That is to say that the first time this runs, function f(*args) is called,
        storing a cache file. The second time and onwards, the resulting
        cached file is read and no execution takes place.

    This assumes each iteration is independent from each other in the list comprehension.

    Parameters
    ----------
    fn : str
        The path and filename.
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)

    Returns
    -------
    res : Any
        The results from f(*args) or from file

    Examples
    --------
    See `turb.utils.umap` for examples.
    """
    if os.path.isfile(fn):
        print("loading file '%s'" % fn)
        return load(fn)
    else:
        um = _parallel_list_comprehension(f, *args)
        # cache result
        dump(um, fn)
        # return
        return um


def umapcc(fn: str, f: Callable, *args):
    """Performs Map comprehension with Caching by Chunks.

    That is to say that the first time this runs, function f(*args) is called,
        storing a cache file. The second time and onwards, the resulting
        cached file is read and no execution takes place.

    Further to this, 'by-chunks' means that each step is stored separately as a file
    and concatenated together at the end.

    Parameters
    ----------
    fn : str
        The path and filename.
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)

    Returns
    -------
    res : Any
        The results from f(*args) or from file

    Examples
    --------
    See `turb.utils.umap` for examples.
    """
    if os.path.isfile(fn):
        print("loading file '%s'" % fn)
        return load(fn)
    else:
        n = len(args[0])
        # run and do chunked caching, using item cache
        um = [cache(add_suf(fn, str(i)), f, *arg) for i, arg in enumerate(zip(*args))]
        # save final version
        dump(um, fn)
        # delete temp versions
        for i in range(n):
            fni = add_suf(fn, str(i))
            if os.path.isfile(fni):
                os.remove(fni)
        # return
        return um


def umappcc(fn: str, f: Callable, *args):
    """Performs Map comprehension with Parallelism and Caching by Chunks.

    That is to say that the first time this runs, function f(*args) is called,
        storing a cache file. The second time and onwards, the resulting
        cached file is read and no execution takes place.

    Further to this, 'by-chunks' means that each step is stored separately as a file
        and concatenated together at the end. This means that if a program stops half way through
        execution, when re-run, it restarts from the last cached element, which is incredibly
        useful during debugging and prototype development.

    This assumes each iteration is independent from each other in the list comprehension.

    Parameters
    ----------
    fn : str
        The path and filename.
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)

    Returns
    -------
    res : Any
        The results from f(*args) or from file

    Examples
    --------
    See `turb.utils.umap` for examples.
    """
    if os.path.isfile(fn):
        print("loading file '%s'" % fn)
        return load(fn)
    else:
        n = len(args[0])
        ncpu = n if n < cpu_count() else (cpu_count() - 1)
        # do list comprehension using parallelism
        um = Parallel(ncpu)(delayed(cache)(add_suf(fn, str(i)), f, *arg) \
                            for i, arg in enumerate(zip(*args)))
        # save final version
        dump(um, fn)
        # delete temp versions
        for i in range(n):
            fni = add_suf(fn, str(i))
            if os.path.isfile(fni):
                os.remove(fni)
        # return
        return um
