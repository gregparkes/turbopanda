#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import warnings


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

            warnings.warn("".join(segments), DeprecationWarning)
            return func(*args, **kwargs)

        return _caching_function

    return decorator
