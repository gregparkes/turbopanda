#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# future imports
from __future__ import absolute_import, division, print_function

import re
# imports
import warnings

__all__ = ('deprecated', 'deprecated_param')


def deprecated(version: str, remove: str = None, instead: str = None, reason: str = None):
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

    def decorator(func):
        """This decorator takes the function.
        """

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

    return decorator


def deprecated_param(version: str, deprecated_args: str, remove: str = None, reason: str = None):
    """A method for handling deprecated arguments within a function.

    deprecated_args can be separated by whitespace, ';', ',', or '|'

    Usage
    -----
    @deprecated_param(version="0.2.3", reason="you may consider using *styles* instead.", deprecated_args='color background_color')
    def paragraph(text, color=None, bg_color=None, styles=None):
        pass
    """

    def _decorator(func):
        def _caching_function(*args, **kwargs):
            # splits arguments into words
            dep_arg = re.findall(r"[\w'_]+", deprecated_args)
            segments = ["Parameter(s) {} deprecated since version {}".format(dep_arg, version)]
            if remove is not None:
                segments.append(", to be removed in version {}".format(remove))
            if reason is not None:
                segments.append("; {}".format(reason))
            # issue warning.
            warnings.warn("".join(segments), FutureWarning)
            return func(*args, **kwargs)

        return _caching_function

    return _decorator
