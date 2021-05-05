#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the caching using joblib library."""

import os
from typing import Callable

# make sure joblib is a requirement for this function.
from turbopanda._dependency import requires


@requires("joblib")
def cache(fn: str, f: Callable, *args, **kwargs):
    """Provides automatic caching for anything using joblib.

    Parameters
    ----------
    fn : str
        The name of the file to cache to, or read from. This is fixed. Include extension
    f : function
        A custom function returning the object to cache
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
        print("loading file '%s'" % fn)
        return joblib.load(fn)
    else:
        print("running chunk '%s'" % fn)
        res = f(*args, **kwargs)
        joblib.dump(res, fn)
        return res
