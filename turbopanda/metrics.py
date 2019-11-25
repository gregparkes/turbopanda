#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:08:52 2019

@author: gparkes
"""

from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd


def z_score(X):
    """
    Accepts X as a pd.Series or pd.DataFrame, np.ndarray
    """
    if isinstance(X, np.ndarrray):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif isinstance(X, pd.Series):
        return (X - X.mean()) / X.std()
    elif isinstance(X, pd.DataFrame):
        return X.transform(lambda x: (x - x.mean()) / x.std())
    else:
        return TypeError("type 'X' not recognised, must be [np.ndarray, pd.Series, pd.DataFrame]")


def spearman(X, y, axis=0):
    return X.aggregate(lambda x: spearmanr(x, y)[0], axis=axis)


def pearson(X, y, axis=0):
    return X.aggregate(lambda x: pearsonr(x, y)[0], axis=axis)
