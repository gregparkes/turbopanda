#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling the edit distance of strings using the levenshtein method."""

import itertools as it
import numpy as np
import pandas as pd

from typing import Optional, List
from turbopanda._dependency import is_tqdm_installed, requires


def _ratio_and_distance(s1: str, s2: str, ratio_calc: bool = True) -> float:
    """levenshtein_ratio_and_distance:
    Calculates levenshtein distance between two strings.
    If ratio_calc = True, the function computes the
    levenshtein distance ratio of similarity between two strings
    For all i and j, distance[i,j] will contain the Levenshtein
    distance between the first i characters of s and the
    first j characters of t
    """
    @jit(nopython=True)
    def _fill_matrix(d, s, t):
        r = d.shape[0]
        c = d.shape[1]
        # fill sides
        for col in range(1, c):
            d[col, 0] = col
        for row in range(1, r):
            d[0, row] = row

        # iterate over and calculate.
        for col in range(1, c):
            for row in range(1, r):
                if s[row - 1] == t[col - 1]:
                    cost = 0  # If the characters are the same in the two strings in a given position [i,j] then cost is 0
                else:
                    # to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                    # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                    cost = 2
                d[row][col] = min(
                    d[row - 1][col] + 1,  # Cost of deletions
                    d[row][col - 1] + 1,  # Cost of insertions
                    d[row - 1][col - 1] + cost,
                )  # Cost of substitutions
        return d

    # Initialize matrix of zeros
    rows = len(s1) + 1
    cols = len(s2) + 1
    distance = np.zeros((rows, cols), dtype=int)
    # fill sides
    # distance[0, :] = np.arange(0, cols, 1, dtype=int)
    # distance[:, 0] = np.arange(0, rows, 1, dtype=int)
    # fill the matrix with values, using JIT.
    distance = _fill_matrix(distance, s1, s2)

    if ratio_calc:
        # Computation of the Levenshtein Distance Ratio
        ratio = ((rows + cols - 2) - distance[-1][-1]) / (rows + cols - 2)
        return ratio
    else:
        return distance[-1][-1]


def _compute_combination(gen):
    # handle if tqdm is installed or not.
    if is_tqdm_installed():
        from tqdm import tqdm
        _generator = tqdm(gen, position=0)
    else:
        _generator = gen

    F = [_ratio_and_distance(a, b, True) for a, b in _generator]
    res = pd.DataFrame(gen, columns=["x", "y"])
    res["L"] = F
    return res


def _mirror_matrix_lev(D):
    X = D.pivot("x", "y", "L").fillna(0.0)
    # add to transpose, remove + 1 from diagonal
    X += X.T - np.eye(X.shape[0])
    return X


@requires("numba")
def levenshtein(
    X: List[str],
    Y: Optional[List[str]] = None,
    as_matrix: bool = False,
    with_replacement: bool = True
) -> pd.DataFrame:
    """Determines the levenshtein matrix distance between every pair of column names.

    If Y is present, performs cartesian_product of X & Y terms, else performs cartesian_product of X & X

    Parameters
    ----------
    X : list of str
        string column names
    Y : list of str, optional
        string column names
    as_matrix : bool, default=False
        If True, returns DataFrame in matrix-form.
    with_replacement : bool, default=True
        If true, returns diagonal elements x_i = x_i.

    Returns
    -------
    L : DataFrame (n x p, 3) or (n, p)
        The levenshtein distance, where n is the number of X elements,
            p is the number of y elements
    """
    from numba import jit

    if as_matrix:
        with_replacement = True

    if Y is None:
        if with_replacement:
            comb = tuple(it.combinations_with_replacement(X, 2))
        else:
            comb = tuple(it.combinations(X, 2))
        res = _compute_combination(comb)
        if as_matrix:
            res = _mirror_matrix_lev(res)
    else:
        prod = tuple(it.product(X, Y))
        res = _compute_combination(prod)
        if as_matrix:
            res = res.pivot("x", "y", "L")

    return res
