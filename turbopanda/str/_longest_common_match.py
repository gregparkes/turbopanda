#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling the longest common substring match problem."""

import itertools as it
from typing import List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, Index

from turbopanda.utils import instance_check, nonnegative, disallow_instance_pair


def _single_common_substring_match(a: str, b: str) -> str:
    """Given two strings, find the longest common substring.

     Also known as the Longest Common Substring problem."""
    from difflib import SequenceMatcher
    match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    # return the longest substring
    if match.size != 0:
        return a[match.a:match.a + match.size]
    else:
        return ""


def common_substrings(a: Union[str, List[str]],
                      b: Optional[Union[str, List[str]]] = None,
                      min_length: int = 2) -> Union[str, Series]:
    """Given at least one pair of strings, find all the best common substring matches.

    By default, if one a is passed, it uses the pairwise combinations between all values in the list,
        otherwise with a + b, the cartesian product of the lists is used.

    Parameters
    ----------
    a : str/list of str
        A word or list of words to find the common substring to
    b : str/list of str, optional
        A word or list of words to find the common substring to
        If None, pairwise combinations in a are used
    min_length: int, default=2
        The minimum accepted length of string for a given pair

    Returns
    -------
    z_up : str/Series
        str returned if (a, b) are strs, else Series of valuecounts
    """

    instance_check(a, (str, list, tuple, Index))
    instance_check(b, (type(None), str, list, tuple, Index))
    nonnegative(min_length, int)
    # prevent a case where a can be a str, b is None
    disallow_instance_pair(a, str, b, type(None))

    filters = ("", "_", "__", "-")
    if isinstance(a, str) and isinstance(b, str):
        return _single_common_substring_match(a, b)
    else:
        if isinstance(a, str):
            a = [a]
        elif isinstance(b, str):
            b = [b]
        # if a is a list of length 1 with no b, return a[0]
        elif isinstance(a, (list, tuple)) and len(a) == 1:
            return a[0]
        # determine pair set.
        if b is None:
            # combination iterator
            pair_groups = it.combinations(a, 2)
        else:
            # cartesian product iterator
            pair_groups = it.product(a, b)
        # generate pairs
        z = [_single_common_substring_match(i, j) for i, j in pair_groups]

        def filter_func(x):
            """Custom function which filters according to tuple and keeps elements >= min length"""
            return (x in filters) or (len(x) < min_length) or (z.count(x) <= 1)
        # filter out naff elements
        z_up = list(it.filterfalse(filter_func, z))
        # save as series valuecounts.
        return Series(z_up).value_counts()
