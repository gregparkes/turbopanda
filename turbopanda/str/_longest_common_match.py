#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling the longest common substring match problem."""

import itertools as it
from typing import List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, Index

from turbopanda._deprecator import deprecated
from turbopanda.utils import pairwise, instance_check, nonnegative, disallow_instance_pair


__all__ = ('common_substrings', 'score_pairwise_common_substring')


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


@deprecated("0.2.6", "0.2.8", reason="No longer necessary")
def score_pairwise_common_substring(pairs, with_plot=True, eps=.001):
    """Provides a default scoring method of choosing good pairs.

    Score determined by:
    .. math:: $O = \alpha L_s + \frac{1}{2} \beta N_s$

    Parameters
    ----------
    pairs : Series
        The chosen pairs from a call to `pairwise_common_substring_matches`
    with_plot : bool
        If True, generates a plot by which the best score options are chosen
    eps : float, optional
        Some value to add to the selection criteria to ensure values on the line are not selected

    Returns
    -------
    ss : list
        A subset of the best pairwise common substrings
    """
    # use pairs

    pl = pairs.index.str.len()
    _y = np.log(pairs)
    nx = [pl.max(), pl.min()]
    ny = [_y.min(), _y.max()]
    z = np.polyfit(nx, ny, 1)
    yp = np.polyval(z, pl)

    if with_plot:
        plt.scatter(pl, _y, marker='x')
        plt.plot(nx, ny, 'k--')
        # outliers
        plt.scatter(pl[_y > yp+eps], _y[_y > yp+eps], marker='o', color='r')
        plt.xlabel("Length of matched string")
        plt.ylabel("Log occurrence of matched string")
        plt.show()

    return _y[_y > yp+eps].index
