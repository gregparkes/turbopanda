#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the map function,
    with parallelism and caching enabled as needed."""

import os
from typing import Callable
import itertools as it
from functools import partial

from ._cache import cache
from ._files import insert_suffix as add_suf
from ._files import check_file_path
from turbopanda._dependency import requires, is_tqdm_installed
from ._tqdm_parallel import TqdmParallel

__all__ = ("zipe", "umap", "umapc", "umapp", "umapcc", "umappc", "umappcc", "umap_validate")


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


def _mini_cache(fn, f, *args):
    from joblib import load, dump

    if os.path.isfile(fn):
        return load(fn)
    else:
        res = f(*args)
        dump(res, fn)
        return res


def _directory_info(fn):
    """Returns the relative file and absolute cache directory information"""
    # create a cache directory in the directory below where to plant the file
    fnspl = fn.rsplit("/", 1)
    if len(fnspl) == 2:
        parent_rel, just_file = fnspl
        parent_abs = os.path.abspath(parent_rel)
    else:
        just_file = fnspl[0]
        parent_rel = "."
        parent_abs = os.getcwd()

    abscachedir = os.path.join(parent_abs, "_tmp_umapcc_")
    relfile = os.path.join(os.path.join(parent_rel, "_tmp_umapcc_"), just_file)
    return relfile, abscachedir


def _create_cache_directory(fn):
    relfile, abscachedir = _directory_info(fn)
    # if it doesn't already exist, create it.
    if not os.path.isdir(abscachedir):
        os.mkdir(abscachedir)

    return relfile, abscachedir


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


def _delete_temps(fn):
    # remove the tree
    import shutil
    try:
        shutil.rmtree(fn)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def _map_comp(f, *args):
    N = len(args)
    narg = len(args[0])
    if is_tqdm_installed():
        from tqdm import tqdm
        _gen = (tqdm(args, position=0, total=narg)
                if N == 0
                else tqdm(it.zip_longest(*args), position=0, total=narg))
    else:
        _gen = args if N == 0 else it.zip_longest(*args)
    # check to make sure every argument is an iterable, and make it one if not
    if N == 0:
        return [f(arg) for arg in _gen]
    else:
        return [f(*arg) for arg in _gen]


def _parallel_list_comprehension(f, *args):
    from joblib import cpu_count, Parallel, delayed

    N = len(args)

    if N == 0:
        return f()
    else:
        if is_tqdm_installed():
            # use tqdm to wrap around our iterable
            _Threaded = TqdmParallel(use_tqdm=True, total=len(args[0]))
        else:
            _Threaded = Parallel

        if N == 1:
            n = len(args[0])
            ncpu = n if n < cpu_count() else (cpu_count() - 1)
            um = _Threaded(ncpu)(delayed(f)(arg) for arg in args[0])
        else:
            ncpu = N if N < cpu_count() else (cpu_count() - 1)
            um = _Threaded(ncpu)(delayed(f)(*arg) for arg in it.zip_longest(*args))
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
    >>> import turbopanda.utils import umap
    >>> umap(lambda x: x**2, [2, 4, 6])
    >>> [4, 16, 36]
    Like the normal mapping, multiple lists map to
    multiple parameters passed to the function:
    >>> umap(lambda x, y: x + y, [2, 4, 6], [1, 2, 4])
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
    # call new cache passing our _map_comp args
    # call function with f as the first arg and all other args after.
    return _mini_cache(fn, _map_comp, f, *args)


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
    # create partial
    return _mini_cache(fn, _parallel_list_comprehension, f, *args)


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
        # check the directory actually exists before continuing
        check_file_path(fn, False, True, 0)

        # use tqdm for display.
        if is_tqdm_installed():
            from tqdm import tqdm
            _generator = enumerate(tqdm(its, position=0, total=n))
        else:
            _generator = enumerate(its)

        # create a cache directory in the directory below where to plant the file
        relfile, abscachedir = _create_cache_directory(fn)
        # run and do chunked caching, using item cache
        # _cache_part = partial(cache, f=f, debug=False, expand_filepath=False)

        um = [
            _mini_cache(add_suf(relfile, str(i)), f, *arg)
            for i, arg in _generator
        ]
        # save final version
        _write_file(um, fn)
        # delete the temp files
        _delete_temps(abscachedir)
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
    from joblib import Parallel, cpu_count, delayed
    if os.path.isfile(fn):
        return _load_file(fn)
    else:
        # pre-compute iterable
        its = list(it.zip_longest(*args))
        n = len(its)
        ncpu = n if n < cpu_count() else (cpu_count() - 1)
        # check the directory actually exists before continuing
        check_file_path(fn, False, True, 0)

        # use tqdm for display.
        if is_tqdm_installed():
            # use our custom tqdm parallel class
            _Threaded = TqdmParallel(use_tqdm=True, total=n)
        else:
            _Threaded = Parallel

        # create a cache directory in the directory below where to plant the file
        relfile, abscachedir = _create_cache_directory(fn)
        # do list comprehension using parallelism
        um = _Threaded(ncpu)(
            delayed(_mini_cache)(add_suf(relfile, str(i)), f, *arg)
            for i, arg in enumerate(its)
        )
        # save final version
        _write_file(um, fn)
        # delete temp versions
        _delete_temps(abscachedir)
        # return
        return um


@requires("joblib")
def umap_validate(fn: str):
    """Validate partial stored cache chunk files into one.

    For use with `umapcc` or `umappcc`.

    Parameters
    ----------
    fn : str
        The path and filename.

    Returns
    -------
    res : Any
        The results from f(*args) from a previous call to `umapcc` or others.

    Examples
    --------
    See `turb.utils.umap` for examples.
    """
    # dig into the cache directory and collect the files, concatenating them together.
    relfile, abscachedir = _directory_info(fn)
    # use glob and get all the files within the directory
    import glob
    sub_file_names = glob.glob(os.path.join(abscachedir, "*"))
    # iterate through the files and load them
    data = [_load_file(s) for s in sub_file_names]
    return data
