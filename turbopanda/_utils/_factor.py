#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods for handling factorizing integers., for plots mainly."""
import itertools as it
import numpy as np
from typing import Tuple, List
from scipy.stats import norm


def _factor(n: int) -> List:
    """
    Collect a list of factors given an integer, excluding 1 and n
    """
    if not isinstance(n, (int, np.int, np.int64)):
        raise TypeError("'n' must be an integer")

    # noinspection PyRedundantParentheses
    def prime_powers(_n):
        """c goes through 2, 3, 5 then the infinite (6n+1, 6n+5) series."""
        for c in it.accumulate(it.chain([2, 1, 2], it.cycle([2, 4]))):
            if c * c > _n:
                break
            if _n % c:
                continue
            d, p = (), c
            while not _n % c:
                _n, p, d = _n // c, p * c, d + (p,)
            yield (d)
        if _n > 1:
            yield ((_n,))

    r = [1]
    for e in prime_powers(n):
        r += [a * b for a in r for b in e]
    return r


def _square_factors(n: int) -> Tuple:
    """
    Given n size, calculate the 'most square' factors of that integer.

    Parameters
    -------
    n : int
        An *even* integer that is factorizable.

    Returns
    -------
    F_t : tuple (2,)
        Two integers representing the most 'square' factors.
    """
    if not isinstance(n, (int, np.int, np.int64)):
        raise TypeError("'n' must of type [int, np.int, np.int64]")
    arr = np.sort(np.asarray(_factor(n)))
    return arr[arr.shape[0] // 2], arr[-1] // arr[arr.shape[0] // 2]


def _diag_factors(n: int) -> Tuple:
    """
    Given n size, calculate the 'most off-edge' factors of that integer.

    Parameters
    -------
    n : int
        An *even* integer that is factorizable.

    Returns
    -------
    F_t : tuple (2,)
        Two integers representing the most 'off-edge' factors.
    """
    if not isinstance(n, (int, np.int, np.int64)):
        raise TypeError("'n' must of type [int, np.int, np.int64]")
    arr = np.sort(np.asarray(_factor(n)))
    return arr[arr.shape[0] // 4], arr[-1] // arr[arr.shape[0] // 4]


def nearest_factors(n: int,
                    shape: str = "square",
                    cutoff: int = 6,
                    search_range: int = 5,
                    w_var: float = 1.5) -> Tuple:
    """Calculate the nearest best factors of a given integer.


    Given n size that may not be even, return the 'most square' factors
    of that integer. Uses square_factors and searches linearly around
    options.

    Parameters
    -------
    n : int
        An integer.
    shape : str
        ['diag' or 'square'], by default uses square factors.
    cutoff : int
        The distance between factors whereby any higher requires a search
    search_range : int
        The number of characters forward to search in
    w_var : float
        The variance applied to the normal distribution weighting

    Returns
    -------
    F_t : tuple (2,)
        Two integers representing the most 'square' factors.
    """
    if shape == "square":
        f_ = _square_factors
    elif shape == "diag":
        f_ = _diag_factors
    else:
        raise ValueError("ftype must be {'diag', 'square'}")

    a, b = f_(n)

    # if our 'best' factors don't cut it...
    if abs(a - b) > cutoff:
        # create Range
        rng = np.arange(n, n + search_range, 1, dtype=np.int64)
        # calculate new scores - using function
        nscores = np.asarray([f_(i) for i in rng])
        # calculate distance
        dist = np.abs(nscores[:, 0] - nscores[:, 1])
        # weight our distances by a normal distribution -
        # we don't want to generate too many plots!
        w_dist = dist * (1. - norm.pdf(rng, n, w_var))
        # calculate new N
        return tuple(nscores[w_dist.argmin()])
    else:
        return a, b
