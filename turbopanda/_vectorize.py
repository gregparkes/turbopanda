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
import os
from tempfile import mkdtemp
import shutil
import itertools as it
import operator
import functools
import numpy as np
from joblib import Parallel, delayed, cpu_count, load, dump
from pandas import Series, Index, DataFrame

from .utils._error_raise import belongs, instance_check
from .utils._cache import cache as utilcache


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


def vectorize(_func=None,
              *,
              parallel=False,
              cache=False,
              return_as="list"):
    """A decorator for making vectorizable function calls.

    Optionally we can parallelize the operation to speed up execution over a long parameter set.

    .. note:: Currently in v0.2.6, with cache=True, parallel=False, as there is no current way we know of
    incorporating persistence with parallelism.

    Keyword Parameters
    ----------
    parallel : bool, default=False
        If True, uses `joblib` to parallelize the list comprehension
    cache : bool, default=False
        If True, creates a cache for each step using `joblib`. If code breaks part way through,
        reloads all steps from the last cache.
    return_as : str, default="list"
        Choose from {'list', 'tuple', 'numpy', 'pandas'}
        Specifies how you want the returned data to look like. Be careful when you use this as you may
        get results you don't expect!
    """
    instance_check((parallel, cache), bool)
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
                # filter out non-dictionaries
                argc = [list(filter(lambda x: not isinstance(x, dict), y)) for y in combined]
                argkc = [_dictchain(list(filter(lambda x: isinstance(x, dict), y))) for y in combined]
                # zip together the  arguments
                pkg = zip(argc, argkc)
                # calculate effective number of CPUs
                n_cpus = len(argc) if len(argc) <= cpu_count() else cpu_count() - 1

                if cache:
                    # create a temporary directory cache
                    savedir = mkdtemp()
                    filename_path = "cache_vectorize_"
                    fn = os.path.join(savedir, filename_path)
                    # parallelize on the cache function
                    if parallel:
                        result = Parallel(n_jobs=n_cpus)(
                            delayed(utilcache)(fn+str(i)+".pkl", f, *arg, **kwarg) for i, (arg, kwarg) in enumerate(pkg)
                        )
                    else:
                        result = [utilcache(fn+str(i)+".pkl", f, *arg, **kwarg) for i, (arg, kwarg) in enumerate(pkg)]
                    # clear temporary directory once completed
                    try:
                        shutil.rmtree(savedir)
                    except OSError:
                        pass  # this can fail with windows
                else:
                    if parallel:
                        # perform parallel operation
                        result = Parallel(n_jobs=n_cpus)(delayed(f)(*arg, **kwarg) for arg, kwarg in pkg)
                    else:
                        result = [f(*arg, **kwarg) for arg, kwarg in pkg]

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
