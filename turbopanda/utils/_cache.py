#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the caching using joblib library."""

import os
from typing import Callable

# make sure joblib is a requirement for this function.
from turbopanda._dependency import requires
from ._files import check_file_path


@requires("joblib")
def cache(fn: str,
          f: Callable,
          debug: bool = True,
          expand_filepath: bool = False,
          *args,
          **kwargs):
    """Provides automatic caching for anything using joblib.

    Parameters
    ----------
    fn : str
        The name of the file to cache to, or read from. This is fixed. Include extension
    f : function
        A custom function returning the object to cache
    debug : bool
        Whether to print statements.
    expand_filepath : bool
        If True, checks the filepath AND adds folders if they dont exist to reach it
    *args : list, optional
        Arguments to pass to f(...)
    **kwargs : dict, optional
        Keyword Arguments to pass to f(...)

    Returns
    -------
    ca : cached element
        This can take many forms, either as list, tuple or dict usually
    """
    from joblib import load, dump

    if os.path.isfile(fn):
        if debug:
            print("loading file '%s'" % fn)
        return load(fn)
    else:
        # firstly check the filepath is real
        if expand_filepath:
            check_file_path(fn, expand_filepath, not expand_filepath, 1 if debug else 0)

        if debug:
            print("running chunk '%s'" % fn)
        res = f(*args, **kwargs)
        if debug:
            # check that the folder path exists.
            print("writing file '%s'" % fn)
        dump(res, fn)
        return res


class Cacheable:
    """A cacheable object for holding file name and callable"""
    def __init__(self, fn):
        self.fn = fn
        self.debug = True
        self.expand_filepath = False

    def __call__(self, f, *args, **kwargs):
        return cache(self.fn, f, self.debug, self.expand_filepath, *args, **kwargs)


class CacheContext:
    """A simple __with__ class for cache contexts using a file name."""
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        return Cacheable(self.fn)

    def __exit__(self, rtype, value, traceback):
        pass
