#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the caching using joblib library."""

import os
from typing import Callable

# make sure joblib is a requirement for this function.
from turbopanda._dependency import requires
from ._files import check_file_path


def _load_file(fn, debug=True):
    import joblib
    # use joblib.load to read in the data
    if debug:
        print("loading file '%s'" % fn)
    return joblib.load(fn)


def _write_file(um, fn, debug=True, create_folders=True):
    import joblib
    # check that the file path is real.
    check_file_path(fn, create_folders, not create_folders, 1 if debug else 0)
    if debug:
        # check that the folder path exists.
        print("writing file '%s'" % fn)
    joblib.dump(um, fn)


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
    import joblib

    if os.path.isfile(fn):
        return _load_file(fn, debug)
    else:
        if debug:
            print("running chunk '%s'" % fn)
        res = f(*args, **kwargs)
        _write_file(res, fn, debug, expand_filepath)
        return res
