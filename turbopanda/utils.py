#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:13:38 2019

@author: gparkes
"""

import numpy as np
import pandas as pd

__all__ = ["is_twotuple","instance_check",
           "chain_intersection","chain_union",
           "is_boolean_series"]


def is_twotuple(L):
    """
    Checks whether an object is a list of (2,) tuples
    """
    if isinstance(L, (list, tuple)):
        for i in L:
            if len(i) != 2:
                raise ValueError("elem i: {} is not of length 2".format(i))
    else:
        raise TypeError("L must be of type [list, tuple]")
    return True


def is_boolean_series(bool_s):
    if not isinstance(bool_s, pd.Series):
        raise TypeError("bool_s must be of type [pd.Series], not {}".format(type(bool_s)))
    if not bool_s.dtype in [bool, np.bool]:
        raise TypeError("bool_s must contain booleans, not type '{}'".format(bool_s.dtype))


def instance_check(a, i):
    if not isinstance(a, i):
        raise TypeError("object '{}' does not belong to type {}".format(a, i))


def chain_intersection(*cgroup):
    """
    Given a group of pandas.Index, perform intersection on A & B & C & .. & K
    """
    mchain = iter(cgroup)
    res = mchain.__next__()
    for m in mchain:
        res = res.intersection(m)
    return res


def chain_union(*cgroup):
    """
    Given a group of pandas.Index, perform union on A | B | C | .. | K
    """
    mchain = iter(cgroup)
    res = mchain.__next__()
    for m in mchain:
        res = res.union(m)
    return res
