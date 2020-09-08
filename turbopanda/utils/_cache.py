#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the caching using joblib library."""

import os
from joblib import load, dump


def cache(fn, f, *args, **kwargs):
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
    if os.path.isfile(fn):
        print("loading file '%s'" % fn)
        return load(fn)
    else:
        print("running chunk '%s'" % fn)
        res = f(*args, **kwargs)
        dump(res, fn)
        return res
