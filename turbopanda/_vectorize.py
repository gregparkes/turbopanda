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

And then in the call, you pass in a Param object for the variable x.
f(Map([1, 2, 3, 4]))

The decorator will search through all the arguments, and any args that are an instance of Param
will be executed in a list-like fashion. If multiple arguments are Param objects, the returned list
is the *product* of all of the combinations.
"""
from collections.abc import Iterable
import itertools as it
import operator
import functools
import numpy as np
from joblib import Parallel, delayed, cpu_count
from pandas import Series, Index, DataFrame

from turbopanda.utils._error_raise import belongs, instance_check


def _expand_dict(k, vs):
    return [{k: v} for v in vs]


def _dictchain(L):
    """Chain together a list of dictionaries into a single dictionary.

    e.g [{'c': 1}, {'d': 3}, {'e': "hi"}] -> {'c': 1, 'd': 3, 'e': "hi"}
    """
    return dict(it.chain(*map(dict.items, L)))


def _any_param(args, kwargs):
    return functools.reduce(
        operator.ior,
        map(lambda o: isinstance(o, Param), it.chain(args, kwargs.values()))
    )


class Param(list):
    """The Param class is responsible for chaining together operations on a single function."""

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


def vectorize(_func=None, *, parallel=False, return_as="list"):
    """A decorator for making vectorizable function calls.

    Optionally we can parallelize the operation to speed up execution over a long parameter set.

    Parameters
    ----------
    parallel : bool
        If True, uses `joblib` to parallelize the list comprehension
    return_as : str, default="list"
        Choose from {'list', 'tuple', 'numpy', 'pandas'}
        Specifies how you want the returned data to look like. Be careful when you use this as you may
        get results you don't expect!
    """
    instance_check(parallel, bool)
    belongs(return_as, ("list", 'tuple', 'numpy', 'pandas'))

    def _decorator_vectorize(f):
        @functools.wraps(f)
        def _wrapped_function(*args, **kwargs):
            # check if Param packaging exists
            if _any_param(args, kwargs):
                # unwrap Vector packaging around arguments
                iterargs = [arg if isinstance(arg, Param) else [arg] for arg in args]
                iterkwargs = [_expand_dict(key, arg) if isinstance(arg, Param) else [{key: arg}] for key, arg in
                              kwargs.items()]
                # get the product of the arguments
                combined = tuple(it.product(*iterargs, *iterkwargs))
                # filter for arguments
                filtered_args = [list(filter(lambda x: not isinstance(x, dict), y)) for y in combined]
                # filter for keyword arguments
                filtered_kwargs = [_dictchain(list(filter(lambda x: isinstance(x, dict), y))) for y in combined]
                # map these arguments and return each type
                # NEW in v0.2.5: parallelize with joblib.
                if parallel:
                    result = Parallel(n_jobs=cpu_count() - 1)(
                        delayed(f)(*arg, **kwarg) for arg, kwarg in zip(filtered_args, filtered_kwargs))
                else:
                    result = [f(*arg, **kwarg) for arg, kwarg in zip(filtered_args, filtered_kwargs)]

                if len(result) == 1:
                    return result[0]
                elif return_as == 'tuple':
                    return tuple(result)
                elif return_as == 'numpy':
                    return np.asarray(result)
                elif return_as == 'pandas':
                    return DataFrame(result).squeeze()
                else:
                    return result
            else:
                return f(*args, **kwargs)

        return _wrapped_function

    if _func is None:
        return _decorator_vectorize
    else:
        return _decorator_vectorize(_func)
