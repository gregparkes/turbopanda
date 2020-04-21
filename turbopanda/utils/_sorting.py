#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for sorting lists/tuples."""

import itertools as it
from typing import Callable, List, Tuple, Union

import numpy as np

from ._error_raise import instance_check

__all__ = ('broadsort')


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
