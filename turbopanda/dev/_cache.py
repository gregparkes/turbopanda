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
from typing import Callable, List, Optional, Tuple, Union

from pandas import DataFrame, concat

# locals
from turbopanda._fileio import read
from turbopanda._metapanda import MetaPanda
from turbopanda.utils import belongs, dictcopy, insert_suffix, instance_check, intersect

__all__ = ("cached", "cache", 'cached_chunk')


def _stack_rows(rows):
    return MetaPanda(concat(rows, axis=0, sort=False, ignore_index=True))


def _set_index_def(df, values=('Unnamed:_0', 'Unnamed: 0', 'colnames', 'index', 'counter')):
    # determine intersection between desired values and columns in df.
    _shared = intersect(df.columns.tolist(), values)
    # set these guys as the new index.
    if _shared.shape[0] > 0:
        df.set_index(_shared.tolist(), inplace=True)


def cached(func: Callable,
           filename: Optional[str] = 'example1.json',
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
    if not callable(func):
        raise ValueError('function is not callable')
    # check that file ends with json or csv
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


def cached_chunk(func: Callable,
                 param_name: str,
                 param_values: Union[List, Tuple],
                 filename: Optional[str] = 'example1.json',
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


def cache(filename: Optional[str] = "example1.json") -> Callable:
    """Provides automatic {.json, .csv} decorator caching for `turb.MetaPanda` or `pd.DataFrame`.

    .. note:: this is a decorator function, not to be called directly.

    Parameters
    --------
    filename : str, optional
        The name of the file to cache to, or read from. This is fixed.
         Accepts {'json', 'csv'} extensions only.

    Warnings
    --------
    ImportWarning
        Returned object from cache isn't of type {pd.DataFrame, MetaPanda}

    Raises
    ------
    TypeError
        `filename` isn't of type `str`
    ValueError
        `filename` extension isn't found in {'json', 'csv'}

    Returns
    -------
    mp : turb.MetaPanda
        The MetaPanda object

    .. note:: custom function must return a `pd.DataFrame` or `turb.MetaPanda` object.

    Examples
    --------
    For example, we call as a decorator to our custom function:
    >>> import turbopanda as turb
    >>> @cache('meta_file.json')
    >>> def f(x):
    ...     return turb.MetaPanda(x)

    See Also
    --------
    cached : Provides automatic {.json, .csv} caching for `turb.MetaPanda` or `pd.DataFrame`.
    """
    # check it is string
    instance_check(filename, str)
    # check that file ends with json or csv
    belongs(filename.rsplit(".")[-1], ("json", "csv"))

    # define decorator
    def decorator(func):
        """Basic decorator."""

        def _caching_function(*args, **kwargs):
            # if we find the file
            if os.path.isfile(filename):
                # read it in
                mdf = read(filename)
                _set_index_def(mdf.df_)
                return mdf
            else:
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
                    warnings.warn("returned object from cache not of type [pd.DataFrame, turb.MetaPanda], not cached",
                                  ImportWarning)
                    return mpf

        return _caching_function

    return decorator
