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
from typing import Optional, Union
import numpy as np
import pandas as pd

# locals
from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda._dependency import is_tqdm_installed
from turbopanda.str import pattern
from turbopanda.utils import belongs, instance_check, is_dataframe_float, bounds_check

from ._bicorr import _partial_bicorr_inner, _bicorr_inner_score, _bicorr_inner_full

__all__ = ("correlate", "row_to_matrix")


def _row_to_matrix(rows: pd.DataFrame, x="x", y="y", piv="r") -> pd.DataFrame:
    """Takes the verbose row output and converts to lighter matrix format."""
    square = rows.pivot_table(index=x, columns=y, values=piv)
    # fillna
    square.fillna(0.0, inplace=True)
    # ready for transpose
    square += square.T
    # eliminate 1 from diag
    square.values[np.diag_indices_from(square.values)] -= 1.0
    return square


""" Helper methods to convert from complex `correlate`
    function to bivariate examples (with partials). """


def _corr_combination(data, comb, niter, covar, cart_z, method, output, verbose):
    # calculate the number of combinations to pass to tqdm to set the progressbar length
    # as comb is an iterable

    # handle if tqdm is installed whether to use progressbar.
    if is_tqdm_installed():
        from tqdm import tqdm
        # wrap the generator around tqdm
        if covar is not None and cart_z:
            _generator = tqdm(it.product(comb, covar), position=0, total=niter)
        else:
            _generator = tqdm(comb, position=0, total=niter)
    else:
        # there is no tqdm
        if covar is not None and cart_z:
            _generator = it.product(comb, covar)
        else:
            _generator = comb

    # with no covariates, simple correlation.
    if covar is None:
        # select appropriate function rho.
        rho = _bicorr_inner_score if output == "score" else _bicorr_inner_full
        # iterate and calculate rho
        result_k = [rho(data[x], data[y], method)
                    for x, y in _generator]
    else:
        # if we cartesian over covariates, produce the product of combinations to each covariate
        if cart_z:
            niter *= len(covar)
            result_k = [
                _partial_bicorr_inner(data, x, y, covar, method=method, output=output)
                for (x, y), z in _generator
            ]
        else:
            # otherwise do all pairwise correlations with a fixed covariate matrix
            result_k = [
                _partial_bicorr_inner(data, x, y, covar, method=method, output=output)
                for x, y, in _generator
            ]

    # we should have a list of dict - assemble the records
    return (
        pd.DataFrame.from_records(result_k)
    )


"""################ PUBLIC FUNCTIONS ################### """


def correlate(
        data: Union[pd.DataFrame, MetaPanda],
        x: Optional[SelectorType] = None,
        y: Optional[SelectorType] = None,
        covar: Optional[SelectorType] = None,
        cartesian_covar: bool = False,
        output: str = "full",
        method: str = "spearman",
        verbose: int = 0,
) -> pd.DataFrame:
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
        If True, and if covar is not None, separates every
            element in covar to individually control for
        using the cartesian product
    output : str, default="full"
        Choose from {'full', 'score'}. Score just returns `r` number.
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

    # correlates X columns against Ybin
    >>> R = turb.correlate(data, x=('X', 'M', 'Y'), y='Ybin')
                    X         M         Y
    Ybin     1.000000  0.392251  0.059771

    # correlates X against Ybin controlling for
    >>> R = turb.correlate(data, x='X', y='Ybin', covar='Y') Y
                     X
    Ybin     -0.149210

    # using a different technique
    >>>  R = turb.correlate(data, method="shepherd")
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
    belongs(
        method,
        (
            "pearson",
            "spearman",
            "kendall",
            "biserial",
            "percbend",
            "shepherd",
            "skipped",
        ),
    )
    belongs(output, ("full","score"))
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
            raise TypeError(
                "`covar` variables in `correlate` all must be of type `float`/continuous."
            )

    # execute various use cases based on the presense of x, y, and covar, respectively.
    if x is None and y is None:
        # here just perform matrix-based correlation
        comb = it.combinations_with_replacement(df.columns, 2)
        niter = (df.columns.shape[0]**2) // 2 + (df.columns.shape[0] // 2)
    elif isinstance(x, (list, tuple, pd.Index)) and y is None:
        # use a subset of x, in union with covar
        comb = it.combinations_with_replacement(x, 2)
        niter = (len(x)**2) // 2 + (len(x) // 2)
    elif isinstance(x, (list, tuple, pd.Index)) and isinstance(y, str):
        # list of x, y str -> matrix-vector cartesian product
        comb = it.product(x, [y])
        niter = len(x)
    elif isinstance(y, (list, tuple, pd.Index)) and isinstance(x, str):
        # list of y, x str -> matrix-vector cartesian product
        comb = it.product(y, [x])
        niter = len(y)
    elif isinstance(x, (list, tuple, pd.Index)) and isinstance(
            y, (list, tuple, pd.Index)
    ):
        # list of x, y -> cartesian product of x: y terms
        comb = it.product(x, y)
        niter = len(x) * len(y)
    else:
        raise ValueError("X: {}; Y: {}; Z: {} combination unknown.".format(x, y, covar))
    # return the combination of these effects.
    return _corr_combination(
        df, comb, niter, covar, cartesian_covar, method, output, verbose
    )


def row_to_matrix(rows: pd.DataFrame, x="x", y="y", piv_value="r"):
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
