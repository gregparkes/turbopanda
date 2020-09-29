#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:08:52 2019

@author: gparkes
"""
# future imports
from __future__ import absolute_import, division, print_function

import itertools as it
# imports
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count

# locals
from turbopanda._metapanda import MetaPanda, SelectorType

from turbopanda.str import pattern
from turbopanda.utils import belongs, instance_check, \
    is_dataframe_float, disallow_instance_pair, bounds_check

from ._bicorr import _partial_bicorr_inner, _bicorr_inner


__all__ = ('correlate', 'row_to_matrix')


def _row_to_matrix(rows: pd.DataFrame, x='x', y='y', piv="r") -> pd.DataFrame:
    """Takes the verbose row output and converts to lighter matrix format."""
    square = rows.pivot_table(index=x, columns=y, values=piv)
    # fillna
    square.fillna(0.0, inplace=True)
    # ready for transpose
    square += square.T
    # eliminate 1 from diag
    square.values[np.diag_indices_from(square.values)] -= 1.
    return square


""" Parallel bicorrelate methods """


def _parallel_bicorr(data, comb, is_parallel, *args, **kwargs):
    # performs optional parallelism on bicorrelations.
    # where comb is ((col_name_x1, col_name_y1), (col_name_x2, col_name_y2), ...)
    if is_parallel:
        return Parallel(cpu_count() - 1)(delayed(_bicorr_inner)(data[x], data[y], *args, **kwargs) for x, y in comb)
    else:
        return [_bicorr_inner(data[x], data[y], *args, **kwargs) for x, y in comb]


def _parallel_partial_bicorr(data, covar, comb, is_parallel, is_cart_z, *args, **kwargs):
    # performs optional parallelism on partial bicorrelations
    # where comb can be ((col_name_x1, col_name_y1), (col_name_x2, col_name_y2), ...)
    # or (((col_name_x1, col_name_y1), z1), ((col_name_x2, col_name_y2), z2), ...)
    if is_cart_z:
        comb_cart = it.product(comb, covar)
        if is_parallel:
            return Parallel(cpu_count()-1)(delayed(_partial_bicorr_inner)(data, x, y, z, *args, **kwargs) for (x, y), z in comb_cart)
        else:
            return [_partial_bicorr_inner(data, x, y, z, *args, **kwargs) for (x, y), z in comb_cart]
    else:
        if is_parallel:
            return Parallel(cpu_count()-1)(delayed(_partial_bicorr_inner)(data, x, y, covar, *args, **kwargs) for x, y in comb)
        else:
            return [_partial_bicorr_inner(data, x, y, covar, *args, **kwargs) for x, y in comb]


""" Helper methods to convert from complex `correlate` function to bivariate examples (with partials). """


def _corr_combination(data, comb, covar, parallel, cart_z, method, verbose):
    # iterate and perform two_variable as before
    if covar is None:
        result_k = _parallel_bicorr(data, comb, parallel, method=method, verbose=verbose)
    else:
        result_k = _parallel_partial_bicorr(data, covar, comb, parallel, cart_z, method=method, verbose=verbose)
    return pd.concat(result_k, axis=0, sort=False).reset_index().rename(columns={"index": "method"})


"""##################### PUBLIC FUNCTIONS ####################################################### """


def correlate(data: Union[pd.DataFrame, MetaPanda],
              x: Optional[SelectorType] = None,
              y: Optional[SelectorType] = None,
              covar: Optional[SelectorType] = None,
              cartesian_covar: bool = False,
              parallel: bool = False,
              method: str = "spearman",
              verbose: int = 0) -> pd.DataFrame:
    """Correlates X and Y together to generate a list of correlations.

    If X/Y are MetaPandas, returns a MetaPanda object, else returns pandas.DataFrame

    Parameters
    ---------
    data : pd.DataFrame / MetaPanda
        The full dataset.
    x : (str, list, tuple, pd.Index), optional
        Subset of input(s) for column names.
            if None, uses the full dataset. Y must be None in this case also.
    y : (str, list, tuple, pd.Index), optional
        Subset of output(s) for column names.
            if None, uses the full dataset (from optional `x` subset)
    covar : (str, list, tuple, pd.Index), optional
        set of covariate(s). Covariates are needed to compute partial correlations.
            If None, uses standard correlation.
    cartesian_covar : bool, default=False
        If True, and if covar is not None, separates every element in covar to individually control for
        using the cartesian product
    parallel : bool, default=False
        If True, computes multiple correlation pairs in parallel using `joblib`. Uses `n_jobs=-2` for default.
    method : str, default="spearman"
        Method to correlate with. Choose from:
            'pearson' : Pearson product-moment correlation
            'spearman' : Spearman rank-order correlation
            'kendall' : Kendall’s tau (ordinal data)
            'biserial' : Biserial correlation (continuous and boolean data only)
            'percbend' : percentage bend correlation (robust)
            'shepherd' : Shepherd's pi correlation (robust Spearman)
            'skipped' : skipped correlation (robust Spearman, requires sklearn)
    verbose : int, default=0
        If > 0, prints out useful debugging messages

    Returns
    -------
    R : pd.DataFrame
        correlation rows (based on pingouin structure)

    Examples
    --------
    >>> import turbopanda as turb
    >>> data = turb.read('example.json')
    >>> R = turb.correlate(data) # uses full dataset
                 X         M         Y      Mbin      Ybin
    X     1.000000  0.392251  0.059771 -0.014405 -0.149210
    M     0.392251  1.000000  0.545618 -0.015622 -0.094309
    Y     0.059771  0.545618  1.000000 -0.007009  0.161334
    Mbin -0.014405 -0.015622 -0.007009  1.000000 -0.076614
    Ybin -0.149210 -0.094309  0.161334 -0.076614  1.000000
    >>> R = turb.correlate(data, x=('X', 'M', 'Y')) # uses subset of dataset
                 X         M         Y
    X     1.000000  0.392251  0.059771
    M     0.392251  1.000000  0.545618
    Y     0.059771  0.545618  1.000000
    >>> R = turb.correlate(data, x=('X', 'M', 'Y'), y='Ybin') # correlates X columns against Ybin
                    X         M         Y
    Ybin     1.000000  0.392251  0.059771
    >>> R = turb.correlate(data, x='X', y='Ybin', covar='Y') # correlates X against Ybin controlling for Y
                     X
    Ybin     -0.149210
    >>>  R = turb.correlate(data, method="shepherd") # using a different technique
                 X         M         Y      Mbin      Ybin
    X     1.000000  0.392251  0.059771 -0.014405 -0.149210
    M     0.392251  1.000000  0.545618 -0.015622 -0.094309
    Y     0.059771  0.545618  1.000000 -0.007009  0.161334
    Mbin -0.014405 -0.015622 -0.007009  1.000000 -0.076614
    Ybin -0.149210 -0.094309  0.161334 -0.076614  1.000000
    """

    # data cannot be NONE
    instance_check(data, (pd.DataFrame, MetaPanda))
    instance_check((x, y, covar), (type(None), str, list, tuple, pd.Index))
    instance_check(cartesian_covar, bool)
    belongs(method, ('pearson', 'spearman', 'kendall', 'biserial', 'percbend', 'shepherd', 'skipped'))
    bounds_check(verbose, 0, 4)

    # downcast to dataframe option
    df = data.df_ if not isinstance(data, pd.DataFrame) else data
    # downcast if list/tuple/pd.index is of length 1
    x = x[0] if (isinstance(x, (tuple, list, pd.Index)) and len(x) == 1) else x
    y = y[0] if (isinstance(y, (tuple, list, pd.Index)) and len(y) == 1) else y

    # convert using `view` if we have string instances.
    if isinstance(x, str):
        x = pattern(x, df.columns)
    if isinstance(y, str):
        y = pattern(y, df.columns)
    if isinstance(covar, str):
        covar = pattern(covar, df.columns)

    # perform a check to make sure every column in `covar` is continuous.
    if covar is not None:
        if not is_dataframe_float(data[covar]):
            raise TypeError("`covar` variables in `correlate` all must be of type `float`/continuous.")

    # execute various use cases based on the presense of x, y, and covar, respectively.
    if x is None and y is None:
        # here just perform matrix-based correlation
        comb = it.combinations_with_replacement(df.columns, 2)
    elif isinstance(x, (list, tuple, pd.Index)) and y is None:
        # use a subset of x, in union with covar
        comb = it.combinations_with_replacement(x, 2)
    elif isinstance(x, (list, tuple, pd.Index)) and isinstance(y, str):
        # list of x, y str -> matrix-vector cartesian product
        comb = it.product(x, [y])
    elif isinstance(y, (list, tuple, pd.Index)) and isinstance(x, str):
        # list of y, x str -> matrix-vector cartesian product
        comb = it.product(y, [x])
    elif isinstance(x, (list, tuple, pd.Index)) and isinstance(y, (list, tuple, pd.Index)):
        # list of x, y -> cartesian product of x: y terms
        comb = it.product(x, y)
    else:
        raise ValueError("X: {}; Y: {}; Z: {} combination unknown.".format(x, y, covar))
    # return the combination of these effects.
    return _corr_combination(df, comb, covar, parallel, cartesian_covar, method, verbose)


def row_to_matrix(rows: pd.DataFrame, x="x", y="y", piv_value='r'):
    """Converts a row-output from `correlate` into matrix form.

    Parameters
    ----------
    rows : pd.DataFrame
        The output from `correlate`
    x : str
        The column to make the index
    y : str
        The column to make the columns
    piv_value : str, optional
        Which parameter to pivot on, by default is `r`, the coefficient.

    Returns
    -------
    m : pd.DataFrame (p, p)
        The correlation matrix
    """
    return _row_to_matrix(rows, x=x, y=y, piv=piv_value)
