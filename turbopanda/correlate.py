#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:08:52 2019

@author: gparkes
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, pointbiserialr

from .metapanda import MetaPanda
from .utils import _cfloat, _intcat


__all__ = ["correlate"]


def _wrap_corr_metapanda(df_corr, pdm):
    C = MetaPanda(df_corr)
    # copy over metadata - dropping columns that aren't present in df_corr
    C.meta_ = pdm.meta_.loc[df_corr.columns]
    # copy over selector
    C._select = pdm.selectors_
    # copy over name
    C.name_ = pdm.name_
    return C


def _handle_debug(debug, xn, yn, r, corr_t):
    if isinstance(debug, bool):
        if debug:
            return "{}:{} ({:0.3f}, {})".format(xn, yn, r, corr_t)


def _r2(x, y):
    r, p = pearsonr(x, y)
    return np.square(r), p


def _accepted_corr():
    return ["spearman", "pearson", "biserial", "r2"]


def _corr_functions():
    return [spearmanr, pearsonr, pointbiserialr, _r2]


def _map_correlate(method):
    mapper=dict(zip(_accepted_corr(), _corr_functions()))
    return mapper[method]


def _corr_two_variables(x, y, method="spearman", debug=False):
    # x and y must share the same index
    if not x.index.equals(y.index):
        raise ValueError("in corr_vector_vector, x.index shape {} does not match y.index shape {}".format(x.index.shape, y.index.shape))
    # get the sharing not-null indices
    shared = x.notnull() & y.notnull()
    # check to make sure shared has non-false values
    if shared.sum() <= 0:
        raise ValueError("No values are shared between {} and {}".format(x.name, y.name))
    # dropna and downcast if possible.
    nx = pd.to_numeric(x[shared], downcast="unsigned", errors="ignore")
    ny = pd.to_numeric(y[shared], downcast="unsigned", errors="ignore")
    # check to see if x or y are constants (all values are the same)
    if nx.unique().shape[0]==1 or ny.unique().shape[0]==1:
        return x.name, y.name, 0., 1., "constant", shared.sum()
    # nx and ny may have new data types
    if nx.dtype in _cfloat() and ny.dtype in _cfloat():
        r = _map_correlate(method)(nx, ny)
        t = method
    elif nx.dtype in _cfloat() and ny.dtype in _intcat():
        r = pointbiserialr(ny, nx)
        t = "biserial"
    elif nx.dtype in _intcat() and ny.dtype in _cfloat():
       r = pointbiserialr(nx, ny)
       t = "biserial"
    elif nx.dtype in _intcat() and ny.dtype in _intcat():
        # both booleans/categorical
        r = spearmanr(nx, ny)
        t = "spearman"
    else:
        raise ValueError("x.dtype '{}' not compatible with y.dtype '{}'".format(x.dtype, y.dtype))

    # handles debug message
    _handle_debug(debug, x.name, y.name, r, t)
    # returns name 1, name2, r, p-val and method used.
    return x.name, y.name, r[0], r[1], t, shared.sum()


def _corr_two_matrix(X, Y, method="spearman", debug=False):
    """
    Computes the correlation between two matrices X and Y of same size.

    No debug of this matrix, since all the names are the same.
    """
    cor = np.zeros((X.shape[1], 2))
    if X.shape != Y.shape:
        raise ValueError("X.shape {} does not match Y.shape {}".format(X.shape, Y.shape))

    for i in range(X.shape[1]):

        cor[i, :] = _corr_two_variables(X.iloc[:, i], Y.iloc[:, i], method, debug=debug)
    return pd.DataFrame(cor, index=X.columns, columns=[method, "p-val"])


def _corr_matrix_singular(X, method="spearman", style="matrix", debug=False):
    """
    Assumes X is of type pandas.DataFrame.

    Drops columns of type 'object' as these cannot be correlated.

    Where:
        - f_i, f_j is continuous: pearson/spearman (method function)
        - f_i is continuous, f_j is categorical: spearman
        - f_i is continuous, f_j is boolean: point biserial correlation
        - f_i, f_j is boolean: pearson product-moment correlation (assuming both are true dichotomous variables)
    """
    # drop 'object' columns
    sel_keep = ~X.dtypes.eq(object)
    if sel_keep.sum() <= 0:
        raise ValueError("there are no non-object-like columns in X")
    elif sel_keep.sum() == 1:
        # perfect correlation with itself
        return (1.0, 0.0)
    elif sel_keep.sum() == 2:
        cols = X.columns[sel_keep]
        return _corr_two_variables(X[cols[0]], X[cols[1]], method, debug=debug)
    else:
        contin_X = X.loc[:, sel_keep]
        # assign zeros
        if style=="matrix":
            R = np.zeros((contin_X.shape[1], contin_X.shape[1]))
        elif style=="rows":
            R = []
        # iterate over i,j pairs
        for i in range(contin_X.shape[1]):
            for j in range(i+1, contin_X.shape[1]):
                # get column types
                if style=="matrix":
                    _, _, R[i, j], _, _, _ = _corr_two_variables(contin_X.iloc[:, i], contin_X.iloc[:, j],
                                                  method=method, debug=debug)
                elif style=="rows":
                    R.append(
                        pd.Series(
                            _corr_two_variables(contin_X.iloc[:, i], contin_X.iloc[:, j],
                                                method=method, debug=debug),
                            index=["column1","column2","r","p-val","rtype","n"]
                        )
                    )

        if style=="matrix":
            # match other side
            R += R.T
            # set diagonal to 1
            R[np.diag_indices(contin_X.shape[1])] = 1.
            # wrap pandas and return
            return pd.DataFrame(R, index=contin_X.columns, columns=contin_X.columns)
        elif style=="rows":
            # concat pd.Series rows together.
            return pd.concat(R, axis=1, sort=False).T


def _corr_matrix_vector(X, y, method="spearman", style="matrix", debug=None):
    if X.shape[0] != y.shape[0]:
        raise ValueError("X.shape {} does not match Y.shape {}".format(X.shape, y.shape))

    if style=="matrix":
        cor = np.zeros((X.shape[1], 2))
        for i in range(X.shape[1]):
            _, _, cor[i, 0], cor[i, 1], _, _ = _corr_two_variables(X.iloc[:, i], y, method, debug=debug)
        return pd.DataFrame(cor, index=X.columns, columns=[method, "p-val"])
    elif style=="rows":
        cor = []
        for i in range(X.shape[1]):
            cor.append(
                pd.Series(
                    _corr_two_variables(X.iloc[:, i], y, method, debug=debug),
                    index=["column1", "column2", "r", "p-val", "rtype", "n"]
                )
            )
        return pd.concat(cor, axis=1, sort=False).T


def correlate(X,
              Y=None,
              method="spearman",
              style="matrix",
              debug=None):
    """
    Correlates X and Y together to generate a correlation matrix.

    If X/Y are MetaPandas, returns a MetaPanda object, else returns pandas.DataFrame

    Parameters
    ---------
    X : pd.Series, pd.DataFrame, MetaPanda
        set of inputs
    Y : pd.Series, pd.DataFrame, MetaPanda, optional
        set of output(s)
    method : str, optional
        Method to correlate with. Choose from
            ['spearman', 'pearson', 'biserial', 'r2']
    style : str, optional
        If 'matrix': returns a pandas.DataFrame square matrix
        If 'rows': returns row-wise (x, y) correlation on each row of pandas.DataFrame (contains more information)
            Note this type only works if X and Y are different, and both are not of type [pd.Series]
    debug : bool, str, optional
        If True, includes a number of print statements.
        If str, assumes it's a file and attempts to create .csv output of statistics.

    Returns
    -------
    R : float/pd.DataFrame/turb.MetaPanda
        correlation matrix/rows
    """
    # select pandas.DataFrame
    NX = X.df_ if isinstance(X, MetaPanda) else X
    NY = Y.df_ if isinstance(Y, MetaPanda) else Y

    # if no Y. do singular matrix.
    if NY is None and isinstance(NX, pd.DataFrame):
        mat = _corr_matrix_singular(NX, method, style=style, debug=debug)
        if isinstance(X, MetaPanda):
            return _wrap_corr_metapanda(mat, X)
        else:
            return mat
    # two series.
    if isinstance(NX, pd.Series) and isinstance(NY, pd.Series):
        return _corr_two_variables(NX, NY, method, debug=debug)
    # one matrix, one vector
    elif isinstance(NX, pd.DataFrame) and isinstance(NY, pd.Series):
        return _corr_matrix_vector(NX, NY, method, style=style, debug=debug)
    # one vector, one matrix
    elif isinstance(NX, pd.Series) and isinstance(NY, pd.DataFrame):
        # swap them over
        return _corr_matrix_vector(NY, NX, method, style=style, debug=debug)
    # two matrices of same shape
    elif isinstance(NX, pd.DataFrame) and isinstance(NY, pd.DataFrame):
        return _corr_two_matrix(NX, NY, method, debug=debug)
    else:
        raise TypeError("X of type [{}], Y of type [{}], cannot compare".format(type(NX), type(NY)))
