#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:08:52 2019

@author: gparkes
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr, pointbiserialr

from .metapanda import MetaPanda
from .utils import _cfloat, _intcat


__all__ = ["correlate", "condition_number"]


def _r2(x, y):
    r, p = pearsonr(x, y)
    return np.square(r), p


def _map_correlate(method):
    if method == "spearman":
        return spearmanr
    elif method == "pearson":
        return pearsonr
    elif method == "r2":
        return _r2
    else:
        raise NotImplementedError


def _mi_matrix(X):
    """
    Computes a mutual information matrix given X only by correlating with every
    other column.
    """
    MI = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            selector = X.iloc[:, i].notnull() & X.iloc[:, j].notnull()
            MI[i, j] = mutual_info_regression(
                    np.atleast_2d(X.loc[selector, X.columns[i]]).T,
                    X.loc[selector, X.columns[j]]
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
    cor = np.zeros((X.shape[1], 2))
    if X.shape != Y.shape:
        raise ValueError("X.shape {} does not match Y.shape {}".format(X.shape, Y.shape))
    for i in range(X.shape[1]):
        # selector = X.iloc[:, i].notnull() & Y.iloc[:, i].notnull()
        cor[i, :] = m_f(X.iloc[:, i], Y.iloc[:, i])
    return pd.DataFrame(cor, index=X.columns, columns=[method, "p-val"])


def _corr_matrix_singular(X, method="spearman"):
    """
    Assumes X is of type MetaPanda to allow for type inference. Assumes e_types exists.

    Drops columns of type 'object' as these cannot be correlated.

    Where:
        - f_i, f_j is continuous: pearson/spearman (method function)
        - f_i is continuous, f_j is categorical: spearman
        - f_i is continuous, f_j is boolean: point biserial correlation
        - f_i, f_j is boolean: pearson product-moment correlation (assuming both are true dichotomous variables)
    """
    # drop 'object' columns
    contig_X = X.compute([("drop", (object,), {})])
    m_func = _map_correlate(method)

    # assign zeros
    R = np.zeros((contig_X.p_, contig_X.p_))

    # iterate over i,j pairs
    for i in range(contig_X.p_):
        for j in range(i+1, contig_X.p_):
            # get column types
            c1 = contig_X.meta_.index[i]; c2 = contig_X.meta_.index[j]
            dt1 = contig_X.meta_.loc[c1, "e_types"]
            dt2 = contig_X.meta_.loc[c2, "e_types"]
            # generate selector
            sel = contig_X.df_.iloc[:, i].notnull() & contig_X.df_.iloc[:, j].notnull()
            # generate subsets - downcasting if necessary
            x = pd.to_numeric(contig_X.df_.loc[sel, c1], downcast="unsigned", errors="ignore")
            y = pd.to_numeric(contig_X.df_.loc[sel, c2], downcast="unsigned", errors="ignore")
            # if both floats, use method correlation
            if dt1 in _cfloat() and dt2 in _cfloat():
                R[i, j] = m_func(x, y)[0]
            elif dt1 in _cfloat() and dt2 in _intcat():
                R[i, j] = pointbiserialr(y, x)[0]
            elif dt1 in _intcat() and dt2 in _cfloat():
                R[i, j] = pointbiserialr(x, y)[0]
            elif dt1 in _intcat() and dt2 in _intcat():
                # both booleans/categorical
                R[i, j] = spearmanr(x, y)[0]
            else:
                raise TypeError("pairing of type [{}] with [{}] not recognized in _corr_matrix_singular".format(dt1, dt2))
    # match other side
    R += R.T
    # set diagonal to 1
    R[np.diag_indices(contig_X.p_)] = 1.
    # wrap pandas and return
    return pd.DataFrame(R, index=contig_X.meta_.index, columns=contig_X.meta_.index)


def _corr_matrix_vector(X, y, method="spearman"):
    m_f = _map_correlate(method)
    cor = np.zeros((X.shape[1],2))
    if X.shape[0] != y.shape[0]:
        raise ValueError("X.shape {} does not match Y.shape {}".format(X.shape, y.shape))
    for i in range(X.shape[1]):
        selector = X.iloc[:, i].notnull() & y.notnull()
        cor[i, :] = m_f(X.loc[selector, X.columns[i]], y.loc[selector])
    return pd.DataFrame(cor, index=X.columns, columns=[method, "p-val"])


def _correlate_options(X, Y, method):
    if isinstance(X, pd.Series) and isinstance(Y, pd.Series):
        return _corr_vector_vector(X, Y, method)
    elif isinstance(X, pd.DataFrame) and isinstance(Y, pd.Series):
        return _corr_matrix_vector(X, Y, method)
    elif isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
        return _corr_two_matrix(X, Y, method)
    else:
        raise TypeError("X of type [{}], Y of type [{}], cannot compare".format(type(X), type(Y)))


def correlate(X,
              Y=None,
              method="spearman"):
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
            ['spearman', 'pearson']

    Returns
    -------
    R : float/pd.DataFrame
        correlation values
    """
    if Y is None and isinstance(X, MetaPanda):
        # use custom correlation.
        return _corr_matrix_singular(X, method)
    elif Y is None and isinstance(X, pd.DataFrame):
        # use default pandas.DataFrame.corr
        return X.corr(method=method)
    else:
        if isinstance(X, MetaPanda) and isinstance(Y, MetaPanda):
            return _correlate_options(X.df_, Y.df_, method)
        elif isinstance(X, MetaPanda) and isinstance(Y, pd.DataFrame):
            return _correlate_options(X.df_, Y, method)
        elif isinstance(X, pd.DataFrame) and isinstance(Y, MetaPanda):
            return _correlate_options(X, Y.df_, method)
        else:
            return _correlate_options(X, Y, method)


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

    X = mdf.compute(pipe, inplace=False).df_
    eigs = np.linalg.eigvals(X.T @ X)
    return np.sqrt(eigs.max() / eigs.min())
