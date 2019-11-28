#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:08:52 2019

@author: gparkes
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr

from .metapanda import MetaPanda


__all__ = ["correlate", "condition_number"]


def _r2(x, y):
    r, p = pearsonr(x, y)
    return np.square(r), p


def _map_correlate(method):
    if method =="spearman":
        return spearmanr
    elif method=="pearson":
        return pearsonr
    elif method=="r2":
        return _r2
    else:
        raise NotImplementedError


def _mi_matrix(X):
    """
    Computes a mutual information matrix given X only by correlating with every
    other column.
    """
    MI = np.zeros((X.shape[1],X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            selector = X.iloc[:, i].notnull() & X.iloc[:, j].notnull()
            MI[i, j] = mutual_info_regression(
                    np.atleast_2d(X.loc[selector,X.columns[i]]).T,
                    X.loc[selector,X.columns[j]]
            )
    # mirror transpose
    MI += MI.T
    # return
    return pd.DataFrame(MI, columns=X.columns, index=X.columns)


def _corr_vector_vector(x, y, method="spearman"):
    m_f = _map_correlate(method)
    return m_f(x, y)


def _corr_two_matrix(X, Y, method="spearman"):
    """
    Computes the correlation between two matrices X and Y of same size.
    """
    m_f = _map_correlate(method)
    cor = np.zeros((X.shape[1],2))
    if X.shape != Y.shape:
        raise ValueError("X.shape {} does not match Y.shape {}".format(X.shape, Y.shape))
    for i in range(X.shape[1]):
        # selector = X.iloc[:, i].notnull() & Y.iloc[:, i].notnull()
        cor[i, :] = m_f(X.iloc[:, i], Y.iloc[:, i])
    return pd.DataFrame(cor, index=X.columns, columns=[method, "p-val"])


def _corr_matrix_vector(X, y, method="spearman"):
    m_f = _map_correlate(method)
    cor = np.zeros((X.shape[1],2))
    if X.shape[0] != y.shape[0]:
        raise ValueError("X.shape {} does not match Y.shape {}".format(X.shape, y.shape))
    for i in range(X.shape[1]):
        selector = X.iloc[:, i].notnull() & y.notnull()
        cor[i, :] = m_f(X.loc[selector, X.columns[i]], y.loc[selector])
    return pd.DataFrame(cor, index=X.columns, columns=[method, "p-val"])


def correlate(X, Y=None, method="spearman"):
    """
    Correlates X and Y together to generate a correlation matrix.

    Parameters
    ---------
    X : pd.Series, pd.DataFrame, MetaPanda
        set of inputs
    Y : None, pd.Series, pd.DataFrame, MetaPanda
        set of output(s)
    method : str
        Method to correlate with. Choose from
            [spearman, pearson, r2]

    Returns
    -------
    R : np.ndarray
        correlation matrix
    """
    # reduce X, Y if metapanda
    if isinstance(X, MetaPanda):
        X = X.df_
    if isinstance(Y, MetaPanda):
        Y = Y.df_

    if Y is None:
        # only handle X
        return X.corr(method=method)
    else:
        if isinstance(X, pd.Series) and isinstance(Y, pd.Series):
            return _corr_vector_vector(X, Y, method)
        elif isinstance(X, pd.DataFrame) and isinstance(Y, pd.Series):
            return _corr_matrix_vector(X, Y, method)
        elif isinstance(Y, np.ndarray) and (Y.ndim == 1):
            return _corr_matrix_vector(X, Y, method)
        elif isinstance(Y, pd.DataFrame):
            return _corr_two_matrix(X, Y, method)
        else:
            raise ValueError("Y not recognized")


def condition_number(mdf):
    """
    Computes the condition number of a dataset.
    """
    # using an external pipe, reduce mdf to something conditionable. maybe?
    pipe = [
        ("drop", (object, "_id$",), {}),
        ("apply", ("dropna",), {}),
        ("transform", (lambda x: x / np.linalg.norm(x, axis=0),), {"whole": True}),
        ("drop", (lambda x: x.count() < x.shape[0],), {}),
    ]

    X = mdf.compute_extern(pipe).df_
    eigs = np.linalg.eigvals(X.T @ X)
    return np.sqrt(eigs.max() / eigs.min())
