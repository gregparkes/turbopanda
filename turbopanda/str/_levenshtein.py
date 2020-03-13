#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operations for handling the edit distance of strings using the levenshtein method."""

import numpy as np
import pandas as pd


def _levenshtein_ratio_and_distance(s, t, ratio_calc=True):
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

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings in a given position [i,j] then cost is 0
            else:
                # to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                cost = 2 if ratio_calc else 1
            distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                     distance[row][col - 1] + 1,  # Cost of insertions
                                     distance[row - 1][col - 1] + cost)  # Cost of substitutions
    if ratio_calc:
        # Computation of the Levenshtein Distance Ratio
        ratio = ((len(s) + len(t)) - distance[-1][-1]) / (len(s) + len(t))
        return ratio
    else:
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[-1][-1])


def _levenshtein_matrix(columns):
    """
    Calculates the pairwise levenshtein distance between every column element i, j
    """
    lev_m = np.zeros((len(columns), len(columns)))
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            lev_m[i, j] = _levenshtein_ratio_and_distance(columns[i], columns[j], True)
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
