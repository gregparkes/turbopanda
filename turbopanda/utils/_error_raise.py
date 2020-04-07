#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raising errors, by doing basic checks."""
from typing import Any, List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

__all__ = ("belongs", "instance_check", "check_list_type", "nonnegative",
           "boolean_series_check", "is_twotuple", "is_iterable")


def belongs(elem: Any, home: Union[List[Any], Tuple[Any, ...]], raised=True):
    """Check whether elem belongs in a list."""
    if elem not in home:
        if raised:
            raise ValueError("element {} is not found in list: {}".format(elem, home))
        else:
            return False
    return True


def nonnegative(a: Union[float, int], raised=True):
    """Check whether value a is nonnegative number. """
    if not isinstance(a, (float, int, np.float, np.int)):
        if raised:
            raise TypeError("object '{}' must be of type [float, int], not type '{}'".format(a, type(a)))
        else:
            return False
    else:
        if a < 0:
            if raised:
                raise AttributeError("object '{}' must be non-negative.".format(a))
            else:
                return False
        else:
            return True


def instance_check(a: object, i: TypeVar, raised=True):
    """Check that a is an instance of type i."""
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


def check_list_type(elems: Tuple, t: TypeVar, raised=True):
    """Checks the type of every element in the list."""
    for i, elem in enumerate(elems):
        if not isinstance(elem, t):
            if raised:
                raise TypeError("type '{}' not found in list at index [{}]".format(t, i))
            else:
                return False
    return True


def boolean_series_check(ser: pd.Series):
    """Check whether ser is full of booleans or not."""
    if not isinstance(ser, pd.Series):
        raise TypeError("bool_s must be of type [pd.Series], not {}".format(type(ser)))
    if ser.dtype not in (bool, np.bool):
        raise TypeError("bool_s must contain booleans, not type '{}'".format(ser.dtype))


def is_twotuple(t: Tuple[Any, Any]):
    """Checks whether an object is a list of (2,) tuples."""
    if isinstance(t, (list, tuple)):
        for i in t:
            if len(i) != 2:
                raise ValueError("elem i: {} is not of length 2".format(i))
    else:
        raise TypeError("L must be of type [list, tuple]")


def is_iterable(e):
    """Determines whether object `e` is an iterable object"""
    try:
        iterator = iter(e)
        return True
    except TypeError:
        return False
