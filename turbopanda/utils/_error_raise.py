#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raising errors, by doing basic checks."""
from typing import Any, List, Tuple, Union, Type

import numpy as np
import pandas as pd
from functools import reduce
import operator
# use scipy.utils.check_array for array_like
from sklearn.utils import check_array

from ._sets import join

__all__ = ("belongs", "instance_check", "disallow_instance_pair",
           "check_list_type", "nonnegative", "bounds_check", "is_twotuple",
           "arrays_equal_size", "arrays_dimension")


def belongs(elem: Any, home: Union[List[Any], Tuple[Any, ...]], raised=True):
    """Check whether elem belongs in a list."""
    if elem not in home:
        if raised:
            raise ValueError("element {} is not found in list: {}".format(elem, home))
        else:
            return False
    return True


def _instance_check_element(a: object, i: Type, raised: bool = True):
    if isinstance(i, str):
        if not hasattr(a, i):
            if raised:
                raise AttributeError("object '{}' does not have attribute '{}'".format(a, i))
            else:
                return False
    elif not isinstance(a, i):
        if raised:
            raise TypeError("object '{}' does not belong to type {}".format(a, i))
        else:
            return False
    elif isinstance(i, (list, tuple)):
        if None in i and a is not None:
            if raised:
                raise TypeError("object '{}' is not of type None".format(a))
            else:
                return False
    return True


def instance_check(a: Union[object, Tuple],
                   i: Union[str, Type, List[Type], Tuple[Type, ...]],
                   raised: bool = True):
    """Check that a is an instance of type i.

    Parameters
    ----------
    a : object or tuple of object
        If tuple, performs check on all in a
    i : type or list of type
        Passed into isinstance which accepts type or list/tuple of type
    raised : bool
        If True, raises error, else just returns false
    """
    if isinstance(a, tuple) and i is not tuple:
        return all([_instance_check_element(x, i) for x in a])
    else:
        return _instance_check_element(a, i, raised)


def nonnegative(a: Union[float, int, Tuple],
                i: Union[Type, Tuple[Type, Type]] = (float, int),
                raised=True):
    """Check whether value a is nonnegative number."""
    instance_check(a, i, raised=raised)
    if isinstance(a, tuple):
        # do nonnegative on all values
        result = any(map(lambda x: x < 0, a))
        if result and raised:
            raise Attribute("Not all values in {} are nonnegative".format(a))
        elif result and not raised:
            return False
        else:
            return True
    else:
        if a < 0 and raised:
            raise Attribute("Not all values in {} are nonnegative".format(a))
        elif a < 0 and not raised:
            return False
        else:
            return True


def bounds_check(x: Union[float, int],
                 lower: Union[float, int],
                 upper: Union[float, int],
                 with_equality: bool = True,
                 raised: bool = True):
    """Checks that x is in a upper/lower bound."""
    instance_check((x, upper, lower), (float, int), raised=raised)
    if with_equality:
        eq = lower <= x <= upper
    else:
        eq = lower < x < upper

    if eq:
        return True
    else:
        if raised:
            raise AttributeError("object bound {} < {} < {} doesn't hold".format(lower, x, upper))
        else:
            return False


def arrays_equal_size(a, b, *arrays):
    """Check that arrays a, b, ...n, are equal dimensions."""
    arrs = tuple(map(np.asarray, join((a, b), arrays)))
    return reduce(operator.add, map(lambda x: x.shape[0], arrs)) // len(arrs) == arrs[0].shape[0]


def arrays_dimension(X: Union[np.ndarray, pd.Series, pd.DataFrame],
                     d: str,
                     raised: bool = True):
    """Check that np.ndarray X is of dimension '1d', '2d', '3d'..."""
    instance_check(X, (np.ndarray, pd.Series, pd.DataFrame))
    belongs(d, ("1d", "2d", "3d"))
    d_int = int(d[0])
    if isinstance(X, pd.Series):
        if d_int != 1:
            if raised:
                raise ValueError("series is %dD, not %dD".format(d_int, X.shape[1]))
            else:
                return False
        else:
            return True
    elif isinstance(X, pd.DataFrame):
        if d_int != 2:
            if raised:
                raise ValueError("dataframe is %dD, not %dD".format(d_int, X.shape[1]))
            else:
                return False
        else:
            return True
    elif X.shape[1] != d_int:
        if raised:
            raise ValueError("array is %dD, not %dD".format(d_int, X.shape[1]))
        else:
            return False
    else:
        return True


def disallow_instance_pair(a: object, i: Type, b: object, j: TypeVar):
    """Defines a pair of objects whereby their types are not allowed as a pair for the function."""
    if instance_check(a, i, raised=False) and instance_check(b, j, raised=False):
        raise TypeError("instance of type '{}' with type '{}' pair disallowed".format(i, j))
    return True


def check_list_type(elems: Tuple, t: Type, raised=True):
    """Checks the type of every element in the list."""
    for i, elem in enumerate(elems):
        if not isinstance(elem, t):
            if raised:
                raise TypeError("type '{}' not found in list at index [{}]".format(t, i))
            else:
                return False
    return True


def is_twotuple(t: Tuple[Any, Any]):
    """Checks whether an object is a list of (2,) tuples."""
    if isinstance(t, (list, tuple)):
        for i in t:
            if len(i) != 2:
                raise ValueError("elem i: {} is not of length 2".format(i))
    else:
        raise TypeError("L must be of type [list, tuple]")
