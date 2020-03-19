#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling the longest common substring match problem."""

import itertools as it
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series

from turbopanda.utils import pairwise

__all__ = ('common_substring_match', 'pairwise_common_substring_matches', 'score_pairwise_common_substring')


def common_substring_match(a: str, b: str) -> str:
    """Given two strings, find the longest common substring.

     Also known as the Longest Common Substring problem."""
    from difflib import SequenceMatcher
    match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    # return the longest substring
    if match.size != 0:
        return a[match.a:match.a + match.size]
    else:
        return ""


def pairwise_common_substring_matches(array: List[str],
                                      filters: Tuple[str, ...] = ('', '_', '__'),
                                      min_length: int = 2) -> Series:
    """
    Given k strings, find the most frequent longest common substring.

    Filtered by:
        - not belonging to a member of `filters`
        - containing more than `min_length` in length
        - having more than 1 occurrence

    Parameters
    ----------
    array : list, tuple, pd.Index
        A list of strings
    filters : tuple
        The character/string elements to filter out
    min_length : int
        The minimum accepted length of string for a given pair

    Returns
    -------
    ser : Series
        The value counts of every pairwise common substring match
    """
    def filter_function(x):
        """Custom function which filters according to tuple and keeps elements >= min length"""
        return (x in filters) or (len(x) < min_length) or (pairs.count(x) <= 1)

    pairs = pairwise(common_substring_match, array)
    # filter out crap elements, such as '', and '_'
    pairs_upd = list(it.filterfalse(filter_function, pairs))
    # save as series and get `value_counts`
    return Series(pairs_upd).value_counts()


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
