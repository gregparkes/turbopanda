#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the map function,
    with parallelism and caching enabled as needed."""

import os
from typing import Callable
import itertools as it

from ._cache import cache
from ._files import insert_suffix as add_suf
from turbopanda._dependency import requires, is_tqdm_installed

__all__ = ("zipe", "umap", "umapc", "umapp", "umapcc", "umappc", "umappcc")


def _load_file(fn, debug=True):
    import joblib
    # use joblib.load to read in the data
    if debug:
        print("loading file '%s'" % fn)
    return joblib.load(fn)


def _write_file(um, fn, debug=True):
    import joblib
    if debug:
        print("writing file '%s'" % fn)
    joblib.dump(um, fn)


def zipe(*args):
    """An extension to the zip() function.

    This function converts single-element arguments
        into the longest-length argument. Note that
        we use `it.zip_longest` and hence differing list
        lengths will introduce None elements.

    Examples
    --------
    This works like a normal zip function when mapping arguments:
    >>> from turbopanda.utils import zipe
    >>> zipe([1, 3, 5], [2, 4, 6])
    >>> [(1, 2), (3, 4), (5, 6)]
    Further to this, single argument lists are converts to the maximum length one by product:
    >>> zipe(1, [4, 6, 8])
    >>> [(1, 4), (1, 6), (1, 8)]
    This extends to more than two arguments, to k, using the longest argument:
    >>> zipe(3, 5, [1, 3], [2, 4, 6])
    >>> [(3, 5, 1, 2), (3, 5, 3, 4), (3, 5, None, 6)]
    """
    if len(args) == 1:
        return args[0]
    else:
        _maxlen = max(map(len, filter(lambda x: isinstance(x, list), args)))

        def _singleton(arg):
            return [arg] * _maxlen if not isinstance(arg, list) else arg

        return list(it.zip_longest(*map(_singleton, args)))


def _delete_temps(fn, n):
    for i in range(n):
        fni = add_suf(fn, str(i))
        if os.path.isfile(fni):
            os.remove(fni)


def _map_comp(f, arg0, *args):
    # check to make sure every argument is an iterable, and make it one if not
    if len(args) == 0:
        return [f(arg) for arg in arg0]
    else:
        return [f(*arg) for arg in it.zip_longest(*args)]


def _parallel_list_comprehension(f, *args):
    import joblib
    if len(args) == 0:
        return f()
    else:
        n = len(args[0])
        ncpu = n if n < joblib.cpu_count() else (joblib.cpu_count() - 1)
        if len(args) == 1:
            um = joblib.Parallel(ncpu)(joblib.delayed(f)(arg) for arg in args[0])
        else:
            um = joblib.Parallel(ncpu)(joblib.delayed(f)(*arg) for arg in it.zip_longest(*args))
        return um


def umap(f: Callable, *args):
    """Performs Map list comprehension.

    Given function f(x) and arguments a, ..., k;
        map f(a_i, ..., k_i), ..., f(a_z, ..., k_z).

    Nearly equivalent to list(map(f, a, ..., k))

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
    Like the normal mapping, multiple lists map to
    multiple parameters passed to the function:
    >>> turb.utils.umap(lambda x, y: x + y, [2, 4, 6], [1, 2, 4])
    >>> [3, 6, 10]
    """
    # check to make sure every argument is an iterable, and make it one if not
    return _map_comp(f, *args)


@requires("joblib")
def umapc(fn: str, f: Callable, *args):
    """Performs Map comprehension with final state Cache.

    That is to say that the first time this runs,
        function f(*args) is called, storing a cache file.
        The second time and onwards, the resulting cached
        file is read and no execution takes place.

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
    return cache(fn, _map_comp, False, f, *args)


@requires("joblib")
def umapp(f: Callable, *args):
    """Performs Map comprehension with Parallelism.

    This assumes each iteration is independent
        from each other in the list comprehension.

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


@requires("joblib")
def umappc(fn: str, f: Callable, *args):
    """Performs Map comprehension with Parallelism and Caching.

    That is to say that the first time this runs, function f(*args) is called,
        storing a cache file. The second time and onwards, the resulting
        cached file is read and no execution takes place.

    This assumes each iteration is independent
        from each other in the list comprehension.

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
        return _load_file(fn)
    else:
        um = _parallel_list_comprehension(f, *args)
        # cache result
        _write_file(um, fn)
        # return
        return um


@requires("joblib")
def umapcc(fn: str, f: Callable, *args):
    """Performs Map comprehension with Caching by Chunks.

    That is to say that the first time this runs, function f(*args) is called,
        storing a cache file. The second time and onwards, the resulting
        cached file is read and no execution takes place.

    Further to this, 'by-chunks' means that each step is stored separately as a file
    and concatenated together at the end. The intermediate caches are removed
    at the end of the process automatically. If the program crashes part-way through
    this, re-running will resume from the last stored chunk.

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
        return _load_file(fn)
    else:
        # pre-compute iterable
        its = list(it.zip_longest(*args))
        n = len(its)
        import shutil
        # use tqdm for display.
        if is_tqdm_installed():
            from tqdm import tqdm
            _generator = enumerate(tqdm(its, position=0, total=n))
        else:
            _generator = enumerate(its)

        # create a cache directory in the directory below where to plant the file
        parent_rel, just_file = fn.rsplit("/", 1)
        parent_abs = os.path.abspath(parent_rel)
        abscachedir = os.path.join(parent_abs, "_tmp_umapcc_")
        relfile = os.path.join(os.path.join(parent_rel, "_tmp_umapcc_"), just_file)

        if not os.path.isdir(abscachedir):
            os.mkdir(abscachedir)

        # run and do chunked caching, using item cache
        um = [
            cache(add_suf(relfile, str(i)), f, False, *arg)
            for i, arg in _generator
        ]
        # save final version
        _write_file(um, fn)
        # delete the temp files
        _delete_temps(relfile, n)
        # return
        return um


@requires("joblib")
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
        return _load_file(fn)
    else:
        n = len(args[0])
        ncpu = n if n < joblib.cpu_count() else (joblib.cpu_count() - 1)
        # do list comprehension using parallelism
        um = joblib.Parallel(ncpu)(
            joblib.delayed(cache)(add_suf(fn, str(i)), f, False, *arg)
            for i, arg in enumerate(it.zip_longest(*args))
        )
        # save final version
        _write_file(um, fn)
        # delete temp versions
        _delete_temps(fn, n)
        # return
        return um
