#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling the edit distance of strings using the levenshtein method."""

import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True)
def _fill_matrix(d, s, t):
    rows = d.shape[0]
    cols = d.shape[1]
    # fill sides
    for col in range(1, cols):
        d[col, 0] = col
    for row in range(1, rows):
        d[0, row] = row

    # iterate over and calculate.
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings in a given position [i,j] then cost is 0
            else:
                # to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                cost = 2
            d[row][col] = min(d[row - 1][col] + 1,  # Cost of deletions
                              d[row][col - 1] + 1,  # Cost of insertions
                              d[row - 1][col - 1] + cost)  # Cost of substitutions
    return d


def _ratio_and_distance(s, t, ratio_calc=True):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)
    # fill sides
    # distance[0, :] = np.arange(0, cols, 1, dtype=int)
    # distance[:, 0] = np.arange(0, rows, 1, dtype=int)
    # fill the matrix with values, using JIT.
    distance = _fill_matrix(distance, s, t)

    if ratio_calc:
        # Computation of the Levenshtein Distance Ratio
        ratio = ((rows + cols - 2) - distance[-1][-1]) / (rows + cols - 2)
        return ratio
    else:
        return distance[-1][-1]


def _levenshtein_matrix(columns):
    """
    Calculates the pairwise levenshtein distance between every column element i, j
    """
    lev_m = np.zeros((len(columns), len(columns)))
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            lev_m[i, j] = _ratio_and_distance(columns[i], columns[j], True)
    lev_m = lev_m + lev_m.T - np.eye(len(columns))
    return pd.DataFrame(lev_m, columns=columns, index=columns)


def levenshtein(columns):
    """Determines the levenshtein matrix distance between every pair of column names.

    Parameters
    ----------
    columns : list-like
        string column names

    Returns
    -------
    L : DataFrame (n, n)
        The levenshtein distance matrix, where n is the number of column elements
    """
    return _levenshtein_matrix(columns)
