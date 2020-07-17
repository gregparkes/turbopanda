#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for sorting lists/tuples."""

import itertools as it
from typing import Callable, List, Tuple, Union
import numpy as np

from ._error_raise import instance_check


def broadsort(a: Union[List, Tuple, np.ndarray]) -> List:
    """Sorts elements of most object types.

    Parameters
    ----------
    a : list/tuple of anything
        Some values to sort.

    Returns
    -------
    a_s : list/tuple
        sorted list of a
    """
    instance_check(a, (list, np.ndarray))

    try:
        a_s = sorted(a)
        return a_s
    except TypeError:
        # if we have objects, try to use the __class__ object to sort them by.
        ss = np.argsort([str(b.__class__) for b in a])
        return np.asarray(a)[ss].tolist()


def unique_ordered(a: np.ndarray) -> np.ndarray:
    """Determines the unique elements in `a` in sorted order."""
    _a = np.asarray(a)
    _, idx = np.unique(_a, return_index=True)
    return _a[np.sort(idx)]


def retuple(tup: List[Tuple]) -> List[Tuple]:
    """Reshapes a list of tuples into a set of arguments that are mappable."""
    # length must be greater than 1
    assert len(tup) > 1, "must be more than one tuple"
    L = len(tup[0])
    # firstly assert that every tuple is of the same length
    assert all(map(lambda t: len(t) == L, tup)), "not all tuples are same length"
    # now reshape according to this length
    return [tuple(map(lambda t: t[i], tup)) for i in range(L)]
