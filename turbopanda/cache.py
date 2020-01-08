#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:29:46 2019

@author: gparkes

Caching functions that take a function which returns a MetaPanda object
and caches it to file, ready to re-install if re-run.
"""
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import os
import warnings
from pandas import DataFrame

# locals
from .metapanda import MetaPanda
from .fileio import read
from .utils import instance_check, belongs


__all__ = ("cached", "cache")


def _set_index_def(df, values=('Unnamed:_0', 'Unnamed: 0')):
    for v in values:
        if v in df.columns:
            df.set_index(v, inplace=True)


def cached(func, filename='example1.json', *args, **kwargs):
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
    *args : list, optional
        Arguments to pass to function(...)
    **kwargs : dict, optional
        Keyword arguments to pass to function (...)

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
    cache : Provides automatic {.json, .csv} caching for `turb.MetaPanda` or `pd.DataFrame`.
    """
    # check it is string
    instance_check(filename, str)
    if not callable(func):
        raise ValueError('function is not callable')
    # check that file ends with json or csv
    belongs(filename.rsplit(".",1)[-1], ("json", "csv"))

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
        elif isinstance(mpf, DataFrame):
            # save
            mpf.to_csv(filename)
            # return as MetaPanda
            return MetaPanda(mpf)
        else:
            warnings.warn("returned object from cache not of type [pd.DataFrame, turb.MetaPanda], not cached",
                          FutureWarning)


def cache(filename="example1.json"):
    """Provides automatic {.json, .csv} caching for `turb.MetaPanda` or `pd.DataFrame`.

    .. note:: this is a decorator function, not to be called directly.

    For example, we call `cached` as a decorator to our custom function:
    >>> import turbopanda as turb
    >>> @cache('meta_file.json')
    >>> def f(x):
    ...     return turb.MetaPanda(x)

    .. note:: custom function must return a `pd.DataFrame` or `turb.MetaPanda` object.

    Parameters
    --------
    filename : str, optional
        The name of the file to cache to, or read from. This is fixed.
         Accepts {'json', 'csv'} extensions only.

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
    mp : turb.MetaPanda
        The MetaPanda object

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
        def _caching_function(*args, **kwargs):
            # if we find the file
            if os.path.isfile(filename):
                # read it in
                return read(filename)
            else:
                # returns MetaPanda or pandas.DataFrame
                mpf = func(*args, **kwargs)
                if isinstance(mpf, MetaPanda):
                    # save file
                    mpf.write(filename)
                elif isinstance(mpf, DataFrame):
                    # save
                    mpf.to_csv(filename)
                else:
                    warnings.warn("returned object from cache not of type [pd.DataFrame, turb.MetaPanda], not cached",
                                  FutureWarning)
                return mpf

        return _caching_function
    return decorator
