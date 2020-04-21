#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides a class template to allow certain functions to have 'vectorizable' parameters.

For example, the following function accepts a float and returns a float:
def f(x: float) -> float:
    return x**2

But what about if you want to essentially vectorize this functionality over a given list:

@turb.mappable
def f(x: float) -> float:
    return x**2

And then in the call, you pass in a Chain object for the variable x.
f(Map([1, 2, 3, 4]))

The decorator will search through all the arguments, and any args that are an instance of Chain
will be executed in a list-like fashion. If multiple arguments are Chain objects, the returned list
is the *product* of all of the combinations.
"""
from collections.abc import Iterable
import itertools as it
import functools
import numpy as np
from joblib import Parallel, delayed, cpu_count
from pandas import Series, Index

from turbopanda.utils import dictchain, belongs, instance_check


def _expand_dict(k, vs):
    return [{k: v} for v in vs]


class Vector(list):
    """The Vector class is responsible for chaining together operations on a single function."""

    def __init__(self, *args):
        """Pass a list-like object as input to be chainable.

        Parameters
        ----------
        s : iterable
            An iterable object to iterate over.
        """
        list.__init__(self, args)

    def __repr__(self):
        return super().__repr__()


def vectorize(_func=None, *, parallel=False):
    """A decorator for making vectorizable function calls.

    Optionally we can parallelize the operation to speed up execution over a long parameter set.
    """
    instance_check(parallel, bool)

    def _decorator_vectorize(f):
        @functools.wraps(f)
        def _wrapped_function(*args, **kwargs):
            # unwrap Vector packaging around arguments
            iterargs = [arg if isinstance(arg, Vector) else [arg] for arg in args]
            iterkwargs = [_expand_dict(key, arg) if isinstance(arg, Vector) else [{key: arg}] for key, arg in kwargs.items()]
            # get the product of the arguments
            combined = tuple(it.product(*iterargs, *iterkwargs))
            # filter for arguments
            filtered_args = [list(filter(lambda x: not isinstance(x, dict), y)) for y in combined]
            # filter for keyword arguments
            filtered_kwargs = [dictchain(list(filter(lambda x: isinstance(x, dict), y))) for y in combined]
            # map these arguments and return each type
            # NEW in v0.2.5: parallelize with joblib.
            if parallel:
                result = Parallel(n_jobs=cpu_count()-1)(delayed(f)(*arg, **kwarg) for arg, kwarg in zip(filtered_args, filtered_kwargs))
            else:
                result = [f(*arg, **kwarg) for arg, kwarg in zip(filtered_args, filtered_kwargs)]
            return result

        return _wrapped_function

    if _func is None:
        return _decorator_vectorize
    else:
        return _decorator_vectorize(_func)
