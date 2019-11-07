#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:29:46 2019

@author: gparkes

Some analysis functions to apply to meta columns.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration

from .utils import chain_union, remove_multi_index, remove_string_spaces
from .selection import categorize


def _levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])


def _levenshtein_matrix(columns):
    """
    Calculates the pairwise levenshtein distance between every column element i, j
    """
    LM = np.zeros((len(columns),len(columns)))
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            LM[i, j] = _levenshtein_ratio_and_distance(columns[i], columns[j], True)
    LM = LM + LM.T - np.eye(len(columns))
    return pd.DataFrame(LM, columns=columns, index=columns)


def agglomerate(columns):
    # calcualte lev distance
    LM = _levenshtein_matrix(columns)
    fa = FeatureAgglomeration(2).fit(LM)
    return fa.labels_


def intersection_grid(indexes):
    """
    Given a list of pd.Index, calculates whether any of the values are shared
    between any of the indexes.
    """
    union_l = []
    for i in range(len(indexes)):
        for j in range(i+1, len(indexes)):
            ints = indexes[i].intersection(indexes[j])
            union_l.append(ints)
    return chain_union(*union_l)


def dataframe_clean(df, cat_thresh=20, def_remove_single_col=True):
    """ Given a raw, unprocessed, dataframe; clean it and prepare it. """
    # convert columns to numeric if possible
    ndf = df.apply(pd.to_numeric, errors="ignore")
    # if multi-column, concatenate to a single column.
    remove_multi_index(ndf)
    # perform categorization
    categorize(ndf, cat_thresh, def_remove_single_col)
    # strip column names of spaces either side
    ndf.columns = ndf.columns.str.strip()
    # strip spaces within text-based features
    remove_string_spaces(ndf)
    # remove spaces/tabs within the column name.
    ndf.columns = ndf.columns.str.replace(" ", "_").str.replace("\t","_").str.replace("-","")
    # sort index
    ndf.sort_index(inplace=True)
    return ndf