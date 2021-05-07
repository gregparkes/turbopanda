#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the caching using joblib library."""

import os
from typing import Callable

# make sure joblib is a requirement for this function.
from turbopanda._dependency import requires


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


@requires("joblib")
def cache(fn: str, f: Callable, debug=True, *args, **kwargs):
    """Provides automatic caching for anything using joblib.

    Parameters
    ----------
    fn : str
        The name of the file to cache to, or read from. This is fixed. Include extension
    f : function
        A custom function returning the object to cache
    debug : bool
        Whether to print statements.
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
        _write_file(res, fn, debug)
        return res
