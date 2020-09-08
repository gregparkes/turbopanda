#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# future imports
from __future__ import absolute_import, division, print_function

import re
import sys
import warnings
import functools

__all__ = ('new', 'unimplemented', 'unstable', 'deprecated', 'deprecated_param')


def new(_func=None, *, version: str = "<unknown>"):
    """A decorator for painting a new function as 'new'."""
    def _decorator_new(func):
        @functools.wraps(func)
        def _wrapper_new(*args, **kwargs):
            warnings.warn(("'{}' is new from {}, "
                           "hence parts of this function may behave unexpectedly")
                          .format(func.__name__, version), FutureWarning)
            return func(*args, **kwargs)

        return _wrapper_new

    if _func is None:
        return _decorator_new
    else:
        return _decorator_new(_func)


def unimplemented(_func=None, *, to_complete: str = "<unknown>"):
    """A decorator for declaring a function written to be incomplete or unimplemented"""

    def _decorator_unimplemented(func):
        @functools.wraps(func)
        def _wrapper_unimplemented(*args, **kwargs):
            warnings.warn(("'{}' is unimplemented, "
                           "parts or whole of this function may not work; "
                           "to be completed in version: {}")
                          .format(func.__name__, to_complete), FutureWarning)
            return func(*args, **kwargs)

        return _wrapper_unimplemented

    if _func is None:
        return _decorator_unimplemented
    else:
        return _decorator_unimplemented(_func)


def unstable(_func=None, *, to_complete: str = "<unknown>"):
    """A decorator for declaring a function written to be incomplete, with possible unstable parts"""

    def _decorator_unimplemented(func):
        @functools.wraps(func)
        def _wrapper_unimplemented(*args, **kwargs):
            warnings.warn(("'{}' may be unstable, "
                           "parts of this function may not work as expected; "
                           "to be completed in version {}")
                          .format(func.__name__, to_complete), FutureWarning)
            return func(*args, **kwargs)

        return _wrapper_unimplemented

    if _func is None:
        return _decorator_unimplemented
    else:
        return _decorator_unimplemented(_func)


def deprecated(version: str,
               remove: str = None,
               instead: str = None,
               reason: str = None):
    """A decorator for deprecating functions.

    Parameters
    ----------
    version : str
        The version the deprecation started
    remove : str, optional
        The version when the function will be removed
    instead : str, optional
        The function name of an alternative function
    reason : str, optional
        A verbose string detailing the reasons behind deprecation

    Usage
    -----
    @deprecated("0.2.4", "0.2.7", reason="function beyond scope of the module", instead=".pipe.zscore")
    """

    def _decorator_deprecate(func):
        @functools.wraps(func)
        def _caching_function(*args, **kwargs):
            segments = ["{} is deprecated since version {}".format(func.__name__, version)]
            if remove is not None:
                segments.append(", to be removed in version {}".format(remove))
            if instead is not None:
                segments.append(", use function '{}' instead".format(instead))
            if reason is not None:
                segments.append(", ({})".format(reason))

            warnings.warn("".join(segments), FutureWarning)
            return func(*args, **kwargs)

        return _caching_function

    return _decorator_deprecate


def deprecated_param(version: str,
                     deprecated_args: str,
                     remove: str = None,
                     reason: str = None):
    """A method for handling deprecated arguments within a function.

    deprecated_args can be separated by whitespace, ';', ',', or '|'

    Usage
    -----
    @deprecated_param(version="0.2.3", reason="you may consider using *styles* instead.", deprecated_args='color background_color')
    def paragraph(text, color=None, bg_color=None, styles=None):
        pass
    """

    def _decorator(func):
        @functools.wraps(func)
        def _caching_function(*args, **kwargs):
            # splits arguments into words
            dep_arg = re.findall(r"[\w'_]+", deprecated_args)
            segments = ["Parameter(s) '{}' deprecated since version {}".format(dep_arg, version)]
            if remove is not None:
                segments.append(", to be removed in version {}".format(remove))
            if reason is not None:
                segments.append("; {}".format(reason))
            # issue warning.
            warnings.warn("".join(segments), FutureWarning)
            return func(*args, **kwargs)

        return _caching_function

    return _decorator
