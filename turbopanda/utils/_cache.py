#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the caching using joblib library."""

import os
from typing import Callable

# make sure joblib is a requirement for this function.
from turbopanda._dependency import requires
from ._files import check_file_path


def _load_file(fn, debug=True):
    from joblib import load
    # use joblib.load to read in the data
    if debug:
        print("loading file '%s'" % fn)
    return load(fn)


def _write_file(um, fn, debug=True, create_folders=True):
    from joblib import dump
    # check that the file path is real.
    check_file_path(fn, create_folders, not create_folders, 1 if debug else 0)
    if debug:
        # check that the folder path exists.
        print("writing file '%s'" % fn)
    dump(um, fn)


def _simple_debug_cache(fn, f, *args):
    from joblib import dump, load
    # perform a cursory check of the files.
    check_file_path(fn, False, True, 1)

    if os.path.isfile(fn):
        print("loading file '%s'" % fn)
        return load(fn)
    else:
        res = f(*args)
        print("writing file '%s'" % fn)
        dump(res, fn)
        return res


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
    if os.path.isfile(fn):
        return _load_file(fn, debug)
    else:
        if debug:
            print("running chunk '%s'" % fn)
        res = f(*args, **kwargs)
        _write_file(res, fn, debug, expand_filepath)
        return res
