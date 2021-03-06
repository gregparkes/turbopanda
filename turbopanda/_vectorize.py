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
"""
from typing import Optional, Callable

import os
import itertools as it
import operator
import numpy as np
from pandas import DataFrame
from functools import wraps, reduce

from .utils._error_raise import belongs, instance_check
from .utils._cache import cache as utilcache
from ._dependency import is_joblib_installed


def _expand_dict(k, vs):
    return [{k: v} for v in vs]


def _dictchain(L):
    """Chain together a list of dictionaries into a single dictionary.

    e.g [{'c': 1}, {'d': 3}, {'e': "hi"}] -> {'c': 1, 'd': 3, 'e': "hi"}
    """
    return dict(it.chain(*map(dict.items, L)))


def _any_param(args, kwargs):
    return reduce(
        operator.ior,
        map(lambda o: isinstance(o, Param), it.chain(args, kwargs.values())),
    )


class Param(list):
    """The Param class is responsible for chaining together
    operations on a single function."""

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


def vectorize(
    _func: Optional[Callable] = None,
    *,
    parallel: bool = False,
    cache: bool = False,
    return_as: str = "list"
):
    """A decorator for making vectorizable function calls.

    Optionally we can parallelize the operation to
        speed up execution over a long parameter set.

    .. note:: In v0.2.8, with cache=True, parallel=False,
        as there is no current way we know of
        incorporating persistence with parallelism.

    Keyword Parameters
    ----------
    parallel : bool, default=False
        If True, uses `joblib` to parallelize the list comprehension
    cache : bool, default=False
        If True, creates a cache for each step using `joblib`.
        Handles cases where code stops part-way through.
    return_as : str, default="list"
        Choose from {'list', 'tuple', 'numpy', 'pandas'}
        Specifies how you want the returned data to look like.
    """
    instance_check((parallel, cache), bool)
    belongs(return_as, ("list", "tuple", "numpy", "pandas"))

    # joblib is a requirement if parallel OR cache are True.
    if is_joblib_installed(raise_error=(parallel or cache)):
        from joblib import cpu_count, Parallel, delayed

    # additional imports.
    from tempfile import mkdtemp
    import shutil

    def _decorator_vectorize(f):
        @wraps(f)
        def _wrapped_function(*args, **kwargs):
            # check if Param packaging exists
            if _any_param(args, kwargs):
                # unwrap Vector packaging around arguments
                iterargs = [arg if isinstance(arg, Param) else [arg] for arg in args]
                iterkwargs = [
                    _expand_dict(key, arg) if isinstance(arg, Param) else [{key: arg}]
                    for key, arg in kwargs.items()
                ]
                # get the product of the arguments
                combined = tuple(it.product(*iterargs, *iterkwargs))
                # filter out non-dictionaries
                argc = [
                    list(filter(lambda x: not isinstance(x, dict), y)) for y in combined
                ]
                argkc = [
                    _dictchain(list(filter(lambda x: isinstance(x, dict), y)))
                    for y in combined
                ]
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
                            delayed(utilcache)(fn + str(i) + ".pkl", f, *arg, **kwarg)
                            for i, (arg, kwarg) in enumerate(pkg)
                        )
                    else:
                        result = [
                            utilcache(fn + str(i) + ".pkl", f, *arg, **kwarg)
                            for i, (arg, kwarg) in enumerate(pkg)
                        ]
                    # clear temporary directory once completed
                    try:
                        shutil.rmtree(savedir)
                    except OSError:
                        pass  # this can fail with windows
                else:
                    if parallel:
                        # perform parallel operation
                        result = Parallel(n_jobs=n_cpus)(
                            delayed(f)(*arg, **kwarg) for arg, kwarg in pkg
                        )
                    else:
                        result = [f(*arg, **kwarg) for arg, kwarg in pkg]

                if len(result) == 1:
                    return result[0]
                elif return_as == "tuple":
                    return tuple(result)
                elif return_as == "numpy":
                    return np.asarray(result)
                elif return_as == "pandas":
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
