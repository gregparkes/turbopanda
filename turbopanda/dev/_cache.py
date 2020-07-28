#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:29:46 2019

@author: gparkes

Caching functions that take a function which returns a MetaPanda object
and caches it to file, ready to re-install if re-run.
"""
# future imports
from __future__ import absolute_import, division, print_function

# imports
import os
import warnings
import functools
import joblib
from typing import Callable, List, Optional, Tuple, Union
from pandas import DataFrame, concat
from joblib import Parallel, delayed, cpu_count

# locals
from turbopanda._deprecator import deprecated
from turbopanda._fileio import read
from turbopanda._metapanda import MetaPanda
from turbopanda.utils import belongs, dictcopy, insert_suffix, \
    instance_check, intersect

__all__ = ("cached", "cache", 'cached_chunk')


def _stack_rows(rows: List[DataFrame]) -> "MetaPanda":
    return MetaPanda(concat(rows, axis=0, sort=False, ignore_index=True))


def _set_index_def(df, values=('Unnamed:_0', 'Unnamed: 0', 'colnames', 'index', 'counter')):
    # determine intersection between desired values and columns in df.
    _shared = intersect(df.columns.tolist(), values)
    # set these guys as the new index.
    if _shared.shape[0] > 0:
        df.set_index(_shared.tolist(), inplace=True)


@deprecated("0.2.8", "0.3", instead=".utils.cache")
def cached(func: Callable,
           filename: str = 'example1.json',
           verbose: int = 0,
           *args,
           **kwargs) -> "MetaPanda":
    """Provides automatic {.json, .csv} caching for `turb.MetaPanda` or `pd.DataFrame`.

    .. note:: this is a direct-call cache function. Not cached.

    For example, we call `cached` as a wrapper to our custom function:
    >>> import turbopanda as turb
    >>> def f(x):
    ...     return turb.MetaPanda(x)
    >>> data = cached(f, 'meta_file.json')

    .. note:: custom function must return a `pd.DataFrame` or `turb.MetaPanda` object.

    Parameters
    --------
    func : function
        A custom function returning the pd.DataFrame/MetaPanda
    filename : str, optional
        The name of the file to cache to, or read from. This is fixed.
        Accepts {'json', 'csv'} formats.
    verbose : int, optional
        If > 0, prints out useful information
    *args : list, optional
        Arguments to pass to function(...)
    **kwargs : dict, optional
        Keyword arguments to pass to function(...)

    Warnings
    --------
    FutureWarning
        Returned object from cache isn't of type {pd.DataFrame, MetaPanda}

    Raises
    ------
    TypeError
        `filename` isn't of type `str`
    ValueError
        `filename` extension isn't found in {'json', 'csv'}

    Returns
    -------
    mp : MetaPanda
        The MetaPanda object

    See Also
    --------
    cache : Provides automatic {.json, .csv} decorator caching for `turb.MetaPanda` or `pd.DataFrame`.
    """
    # check it is string
    instance_check(filename, str)
    instance_check(verbose, int)
    instance_check(func, "__call__")

    # check that file ends with json or csv
    file_ext = filename.rsplit(".")[-1]
    belongs(filename.rsplit(".", 1)[-1], ("json", "csv"))

    if os.path.isfile(filename):
        if verbose > 0:
            print("reading in cached file: {}".format(filename))
        # read it in
        mdf = read(filename)
        _set_index_def(mdf.df_)
        return mdf
    else:
        if verbose > 0:
            print("running function '{}' for cache".format(func.__name__))
        # returns MetaPanda or pandas.DataFrame
        mpf = func(*args, **kwargs)
        if isinstance(mpf, MetaPanda):
            # save file
            mpf.write(filename)
            return mpf
        elif isinstance(mpf, DataFrame):
            # save - bumping index into the file.
            mpf.reset_index().to_csv(filename, index=None)
            return MetaPanda(mpf)
        else:
            if verbose > 0:
                print("returned object from cache not of type [DataFrame, MetaPanda], not cached")
            return mpf


@deprecated("0.2.8", "0.3", instead=".utils.umappc",
            reason="The 'umap' suite performs this in a cleaner, more succinct way")
def cached_chunk(func: Callable,
                 param_name: str,
                 param_values: Union[List, Tuple],
                 parallel: bool = True,
                 filename: str = 'example1.json',
                 verbose: int = 0,
                 *args, **kwargs) -> "MetaPanda":
    """Provides chunked automatic {.json, .csv} caching for `turb.MetaPanda` or `pd.DataFrame`.

    .. note:: custom function must return a `pd.DataFrame` or `turb.MetaPanda` object.

    Parameters
    --------
    func : function
        A custom function returning the pd.DataFrame/MetaPanda
    param_name : str
        The keyword name of the parameter in question to iterate over
    param_values : list/tuple of something
        The values associated with the parameter to iterate over
    parallel : bool, default=True
        Determines whether to use `joblib` to compute independent chunks in parallel or not
    filename : str, optional
        The name of the file to cache to, or read from. This is fixed.
        Accepts {'json', 'csv'} formats.
    verbose : int, optional
        If > 0, prints out useful information
    *args : list, optional
        Arguments to pass to function(...)
    **kwargs : dict, optional
        Keyword arguments to pass to function(...)

    Warnings
    --------
    FutureWarning
        Returned object from cache isn't of type {pd.DataFrame, MetaPanda}

    Raises
    ------
    TypeError
        `filename` isn't of type `str`
    ValueError
        `filename` extension isn't found in {'json', 'csv'}

    Returns
    -------
    mp : MetaPanda
        The MetaPanda object

    See Also
    --------
    cached : Provides automatic {.json, .csv} caching for `turb.MetaPanda` or `pd.DataFrame`.
    """
    # check it is string
    instance_check(filename, str)
    instance_check(param_name, str)
    instance_check(param_values, (list, tuple, dict))

    if not callable(func):
        raise ValueError('function is not callable')
    # check that file ends with json or csv
    belongs(filename.rsplit(".", 1)[-1], ("json", "csv"))

    # if the final file exists, perform as normal.
    if os.path.isfile(filename):
        if verbose > 0:
            print("reading in cached file: {}".format(filename))
        # read it in
        mdf = read(filename)
        _set_index_def(mdf.df_)
        return mdf
    else:
        # create a bunch of chunks by repeatedly calling cache.
        if parallel:
            _mdf_chunks = joblib.Parallel(joblib.cpu_count())(
                joblib.delayed(cached)(func, insert_suffix(filename, "_chunk%d" % i),
                                       verbose=verbose, *args, **dictcopy(kwargs, {param_name: chunk})).df_
                for i, chunk in enumerate(param_values)
            )
        else:
            _mdf_chunks = [
                cached(func,
                       insert_suffix(filename, "_chunk%d" % i),
                       verbose=verbose,
                       *args, **dictcopy(kwargs, {param_name: chunk})).df_
                for i, chunk in enumerate(param_values)
            ]
        # join together the chunks
        mpf = _stack_rows(_mdf_chunks)
        # save file - return type must be a MetaPanda or error occurs!
        mpf.write(filename)
        # now delete the 'chunked' files.
        for i in range(len(param_values)):
            os.remove(insert_suffix(filename, "_chunk%d" % i))

        return mpf


def cache(_func=None, *,
          filename: str = "example1.pkl",
          compress=0,
          return_as="MetaPanda") -> Callable:
    """Provides automatic decorator caching for objects.

    Especially compatible with `turb.MetaPanda` or `pd.DataFrame`.

    .. note:: this is a decorator function, not to be called directly.

    Parameters
    ----------
    filename : str, optional
        The name of the file to cache to, or read from. This is fixed.
         Accepts {'json', 'csv', 'pkl'} extensions only.
    compress : int [0-9] or 2-tuple, optional
        Optional compression level for the data. 0 or False is no compression.
        Higher value means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        See the notes for more details.
        If compress is True, the compression level used is 3.
        If compress is a 2-tuple, the first element must correspond to a string
        between supported compressors (e.g 'zlib', 'gzip', 'bz2', 'lzma'
        'xz'), the second element must be an integer from 0 to 9, corresponding
        to the compression level.
    return_as : str, default="MetaPanda"
        Accepts {'pandas', 'MetaPanda'}
        Only applies if filename is "csv" or "json". Attempts to cast the return object
        as something palatable to the user.

    Warnings
    --------
    ImportWarning
        Returned object from cache isn't of type {pd.DataFrame, MetaPanda}

    Raises
    ------
    TypeError
        `filename` isn't of type `str`
    ValueError
        `filename` extension isn't found in {'json', 'csv', 'pkl'}

    Returns
    -------
    mp : turb.MetaPanda / object
        The MetaPanda object if {'csv' or 'json'}, otherwise uses
        serialized pickling which can return an arbritrary object.

    Examples
    --------
    For example, we call as a decorator to our custom function:
    >>> from turbopanda import cache
    >>> @cache('meta_file.json')
    >>> def f(x):
    ...     return turb.MetaPanda(x)
    These also work with numpy arrays or python objects by using `joblib`:
    >>> from turbopanda import cache
    >>> @cache("meta.pkl")
    >>> def g(x):
    ...     return [1, 2, [3, 4], {"hi":"moto"}]
    """
    # check it is string
    instance_check(filename, str)
    file_ext = filename.rsplit(".")[-1]
    # check that file ends with json or csv
    belongs(file_ext, ("json", "csv", "pkl"))

    # define decorator
    def _decorator_cache(func):
        """Basic decorator."""

        @functools.wraps(func)
        def _wrapper_cache(*args, **kwargs):
            # if we find the file
            if os.path.isfile(filename):
                # if its .csv or .json, use `read`
                if file_ext in ('json', 'csv'):
                    # read it in
                    mdf = read(filename)
                    _set_index_def(mdf.df_)
                    if return_as == "MetaPanda":
                        return mdf
                    else:
                        return mdf.df_
                else:
                    mdf = joblib.load(filename)
                    return mdf
            else:
                # returns MetaPanda or pandas.DataFrame
                mpf = func(*args, **kwargs)
                if isinstance(mpf, MetaPanda):
                    # save file
                    mpf.write(filename)
                    if return_as == "MetaPanda":
                        return mpf
                    else:
                        return mpf.df_
                elif isinstance(mpf, DataFrame):
                    # save - bumping index into the file.
                    mpf.reset_index().to_csv(filename, index=None)
                    if return_as == "MetaPanda":
                        return MetaPanda(mpf)
                    else:
                        return mpf
                else:
                    # attempt to use joblib to dump
                    joblib.dump(mpf, filename, compress=compress)
                    return mpf

        return _wrapper_cache

    if _func is None:
        return _decorator_cache
    else:
        return _decorator_cache(_func)
