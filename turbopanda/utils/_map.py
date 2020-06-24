#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the map function, with parallelism and caching enabled as needed."""

import os
from typing import Callable
from joblib import load, dump, delayed, Parallel, cpu_count

from ._files import insert_suffix as add_suf


__all__ = ('umap', 'umapc', 'umapp', 'umapcc', 'umappc', 'umappcc')


def _item_cache(fn, f, *args):
    if os.path.isfile(fn):
        print("loading file '%s'" % fn)
        return load(fn)
    else:
        print("running chunk '%s'" % fn)
        res = f(*args)
        dump(res, fn)
        return res


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
    """Performs Map comprehension as arguments a, b, ..., k to function f(a, b, ..., k)

    Given function f(x) and arguments a, ..., k; map f(a), ..., f(k).

    Parameters
    ----------
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)

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
    return list(map(f, *args))


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
        um = list(map(f, *args))
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
        um = [_item_cache(add_suf(fn, str(i)), f, *arg) for i, arg in enumerate(zip(*args))]
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
        um = Parallel(ncpu)(delayed(_item_cache)(add_suf(fn, str(i)), f, *arg) \
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
