#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:08:52 2019

@author: gparkes
"""
# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import warnings
import numpy as np
import itertools as it
import pandas as pd
from scipy.stats import pearsonr, spearmanr, pointbiserialr

# locals
from .metapanda import MetaPanda
from .utils import c_float, intcat, instance_check
from .deprecator import deprecated


__all__ = ("correlate", "corr_long_to_short")


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
    mapper = dict(zip(_accepted_corr(), _corr_functions()))
    return mapper[method]


def _row_to_matrix(rows):
    """
    Takes the verbose row output and converts to lighter matrix format.
    """
    square = rows.pivot("column1", "column2", "r")
    # fillna
    square.fillna(0.0, inplace=True)
    # ready for transpose
    square += square.T
    # eliminate 1 from diag
    square.values[np.diag_indices_from(square.values)] -= 1.
    return square


def _corr_two_variables(x, y, method="spearman", debug=False):
    # returns name 1, name2, r, p-val and method used.
    df_cols = ["column1", "column2", "r", "p-val", "rtype", "n"]
    # x and y must share the same index
    if not x.index.equals(y.index):
        raise ValueError("in corr_vector_vector, x.index shape {} does not match y.index shape {}".format(
            x.index.shape, y.index.shape))
    # get the sharing not-null indices
    shared = x.notnull() & y.notnull()
    # check to make sure shared has non-false values
    if shared.sum() <= 0:
        raise ValueError("No values are shared between {} and {}".format(x.name, y.name))
    # dropna and downcast if possible.
    nx = pd.to_numeric(x[shared], downcast="unsigned", errors="ignore")
    ny = pd.to_numeric(y[shared], downcast="unsigned", errors="ignore")
    # check to see if x or y are constants (all values are the same)
    if nx.unique().shape[0] == 1 or ny.unique().shape[0] == 1:
        return dict(zip(
            df_cols,
            [x.name, y.name, 0., 1., "constant", shared.sum()]
        ))
    # nx and ny may have new data types
    if nx.dtype in c_float() and ny.dtype in c_float():
        r = _map_correlate(method)(nx, ny)
        t = method
    elif nx.dtype in c_float() and ny.dtype in intcat():
        r = pointbiserialr(ny, nx)
        t = "biserial"
    elif nx.dtype in intcat() and ny.dtype in c_float():
        r = pointbiserialr(nx, ny)
        t = "biserial"
    elif nx.dtype in intcat() and ny.dtype in intcat():
        # both booleans/categorical
        r = spearmanr(nx, ny)
        t = "spearman"
    else:
        raise ValueError("x.dtype '{}' not compatible with y.dtype '{}'".format(nx.dtype, ny.dtype))

    # handles debug message
    _handle_debug(debug, x.name, y.name, r, t)
    # dictzip and return
    return dict(zip(df_cols, [x.name, y.name, r[0], r[1], t, shared.sum()]))


def _corr_two_matrix_same(x, y, method="spearman", debug=False):
    """
    Computes the correlation between two matrices X and Y of same size.

    No debug of this matrix, since all the names are the same.
    """
    cor = np.zeros((x.shape[1], 2))
    if x.shape != y.shape:
        raise ValueError("X.shape {} does not match Y.shape {}".format(x.shape, y.shape))

    for i in range(x.shape[1]):
        cor[i, :] = _corr_two_variables(x.iloc[:, i], y.iloc[:, i], method, debug=debug)
    return pd.DataFrame(cor, index=x.columns, columns=[method, "p-val"])


def _corr_two_matrix_diff(x, y, method='spearman', style='matrix', debug=False):
    """
    Computes the correlation between two matrices X and Y of different columns lengths.

    Essentially computes multiple iterations of corr_matrix_vector.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("X.rows: {} does not match Y.rows: {}".format(x.shape[0], y.shape[1]))
    # create combinations
    comb = list(it.product(x.columns.tolist(), y.columns.tolist()))
    # iterate and perform two_variable as before
    DATA = pd.concat([
        pd.Series(_corr_two_variables(x[x], y[y], method, debug=debug)) for (x, y) in comb
    ], axis=1, sort=False).T

    return DATA if style == 'rows' else _row_to_matrix(DATA)


def _corr_matrix_singular(x, method="spearman", style="matrix", debug=False):
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
    sel_keep = ~x.dtypes.eq(object)
    if sel_keep.sum() <= 0:
        raise ValueError("there are no non-object-like columns in X")
    elif sel_keep.sum() == 1:
        # perfect correlation with itself
        return 1.0, 0.0
    elif sel_keep.sum() == 2:
        cols = x.columns[sel_keep]
        return pd.Series(_corr_two_variables(x[cols[0]], x[cols[1]], method, debug=debug))
    else:
        contin_X = x.loc[:, sel_keep]
        # assign zeros/empty
        R = []
        # iterate over i,j pairs
        for i in range(contin_X.shape[1]):
            for j in range(i, contin_X.shape[1]):
                R.append(
                    pd.Series(
                        _corr_two_variables(contin_X.iloc[:, i], contin_X.iloc[:, j],
                                            method=method, debug=debug)
                    )
                )
        DATA = pd.concat(R, axis=1, sort=False).T

        return DATA if style == 'rows' else _row_to_matrix(DATA)


def _corr_matrix_vector(x, y, method="spearman", style="matrix", debug=None):
    if x.shape[0] != y.shape[0]:
        raise ValueError("X.shape {} does not match Y.shape {}".format(x.shape, y.shape))

    DATA = pd.concat(
        [pd.Series(_corr_two_variables(x.iloc[:, i], y, method, debug=debug)) for i in range(x.shape[1])],
        axis=1, sort=False).T

    return DATA if style == 'rows' else _row_to_matrix(DATA)


##################### PUBLIC FUNCTIONS #######################################################


def correlate(x, y=None, data=None, method="spearman", style="rows", debug=None):
    """
    Correlates X and Y together to generate a correlation matrix.

    If X/Y are MetaPandas, returns a MetaPanda object, else returns pandas.DataFrame

    Parameters
    ---------
    x : (str, list, tuple, pd.Index) / pd.Series, pd.DataFrame, MetaPanda
        set of input(s). If data is non-None, x must be in the first group. 'str' inputs
        must accompany a MetaPanda.
    y : (str, list, tuple, pd.Index) / pd.Series, pd.DataFrame, MetaPanda, optional
        set of output(s). If data is non-None, y must be in the first group or None. 'str'
        inputs must accompany a MetaPanda.
    data : pd.DataFrame, MetaPanda, optional
        If this is None, x must contain the data, else
        if this is not None, x and/or y must contain lists of column names
        (as tuple, list or pd.Index)
    method : str, optional
        Method to correlate with. Choose from {'spearman', 'pearson', 'biserial', 'r2'}
    style : str, optional
        Choose from {'matrix', 'rows'}
        If 'matrix': returns a pandas.DataFrame square matrix
        If 'rows': returns row-wise (x, y) correlation on each row of pandas.DataFrame (contains more information)
            Note this type only works if X and Y are different, and both are not of type {pd.Series}
    debug : bool, str, optional
        If True, includes a number of print statements.
        If str, assumes it's a file and attempts to create .csv output of statistics.

    Returns
    -------
    R : float/pd.DataFrame/turb.MetaPanda
        correlation matrix/rows
    """
    # check for data
    if data is None:
        # assert that X is pd.Series, pd.DataFrame, MetaPanda
        instance_check(x, (pd.Series, pd.DataFrame, MetaPanda))
        if y is not None:
            instance_check(y, (pd.Series, pd.DataFrame, MetaPanda))
        # select pandas.DataFrame
        NX = x.df_ if isinstance(x, MetaPanda) else x
        NY = y.df_ if isinstance(y, MetaPanda) else y
    else:
        instance_check(data, (pd.DataFrame, MetaPanda))
        instance_check(x, (str, list, tuple, pd.Index))
        if y is not None:
            instance_check(y, (str, list, tuple, pd.Index))
        # fetch columns
        X_c = data.view(x) if (isinstance(x, str) and isinstance(data, MetaPanda)) else x
        Y_c = data.view(y) if (isinstance(y, str) and isinstance(data, MetaPanda)) else y
        # fetch X matrix
        NX = data.df_[X_c] if isinstance(data, MetaPanda) else data[X_c]
        if y is None:
            NY = None
        else:
            NY = data.df_[Y_c] if isinstance(data, MetaPanda) else data[Y_c]

    """ Handle different use cases....
        1. Y is None, and we have a DataFrame
        2. X and Y are series
        3. X is DataFrame and Y is seres
        4. X and Y are DataFrame of same size
    """

    # if no Y. do singular matrix.
    if NY is None and isinstance(NX, pd.DataFrame):
        mat = _corr_matrix_singular(NX, method, style=style, debug=debug)
        if isinstance(x, MetaPanda) and style == "matrix":
            return _wrap_corr_metapanda(mat, x)
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
        if NX.shape[1] == NY.shape[1]:
            return _corr_two_matrix_same(NX, NY, method, debug=debug)
        else:
            return _corr_two_matrix_diff(NX, NY, method, style=style, debug=debug)
    else:
        raise TypeError("X of type [{}], Y of type [{}], cannot compare".format(type(NX), type(NY)))


@deprecated('0.1.8', '0.2.0', 'This method is no longer relevant')
def corr_long_to_short(df):
    """
    Converts a long row-wise form of correlation into matrix.

    .. TODO deprecated:: 0.1.8
        This function will be removed in 0.2.0

    Parameters
    ----------
    df : pandas.DataFrame
        The long-form dataframe.
    """
    return _row_to_matrix(df)