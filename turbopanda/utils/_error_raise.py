#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raising errors, by doing basic checks."""
import numpy as np
import pandas as pd
from typing import List, Union, Any, Tuple, TypeVar


__all__ = ("belongs", "instance_check", "check_list_type",
           "boolean_series_check", "is_twotuple", "is_iterable")


def belongs(elem: Any, home: Union[List[Any], Tuple[Any, ...]]):
    """Check whether elem belongs in a list."""
    if elem not in home:
        raise ValueError("element {} is not found in list: {}".format(elem, home))


def instance_check(a: object, i: TypeVar):
    """Check that a is an instance of type i."""
    if isinstance(i, str):
        if not hasattr(a, i):
            raise AttributeError("object '{}' does not have attribute '{}'".format(a, i))
    elif not isinstance(a, i):
        raise TypeError("object '{}' does not belong to type {}".format(a, i))
    elif isinstance(i, (list, tuple)):
        if None in i and a is not None:
            raise TypeError("object '{}' is not of type None".format(a))


def check_list_type(elems: Tuple, t: TypeVar):
    """Checks the type of every element in the list."""
    for i, elem in enumerate(elems):
        if not isinstance(elem, t):
            raise TypeError("type '{}' not found in list at index [{}]".format(t, i))
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
