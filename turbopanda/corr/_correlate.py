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
from scipy import stats
from joblib import Parallel, delayed, cpu_count

# locals
from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.stats._lmfast import lm
from turbopanda.str import pattern
from turbopanda.utils import belongs, difference, instance_check, \
    is_column_boolean, is_column_float, remove_na, union, is_dataframe_float, \
    disallow_instance_pair, bounds_check

from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr
from ._corr_metrics import percbend, shepherd, skipped
from ._stats_extra import compute_esci, power_corr

# user define dataset type
DataSetType = Union[pd.Series, pd.DataFrame, MetaPanda]

__all__ = ('correlate', 'bicorr', 'partial_bicorr', 'row_to_matrix')


"""Methods to handle continuous-continuous, continuous-boolean and boolean-boolean cases of correlation. """


def _both_continuous(x, y):
    return is_column_float(x) and is_column_float(y)


def _continuous_bool(x, y):
    return is_column_float(x) and is_column_boolean(y)


def _bool_continuous(x, y):
    return is_column_boolean(x) and is_column_float(y)


def _boolbool(x, y):
    return is_column_boolean(x) and is_column_boolean(y)


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


""" Internal bivariate methods. """


def _bicorr_inner(x, y, tail='two-sided', method='spearman', verbose=0):
    """Internal method for bicorrelation here"""
    # convert to numpy
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Remove NA
    x_arr, y_arr = remove_na(x_arr, y_arr, paired=True)
    nx = x_arr.size

    outliers = []
    # Compute correlation coefficient
    if _both_continuous(x, y):
        # use method
        if method == 'pearson':
            r, pval = pearsonr(x_arr, y_arr)
        elif method == 'spearman':
            r, pval = spearmanr(x_arr, y_arr)
        elif method == 'kendall':
            r, pval = kendalltau(x_arr, y_arr)
        elif method == 'percbend':
            r, pval = percbend(x_arr, y_arr)
        elif method == 'shepherd':
            r, pval, outliers = shepherd(x_arr, y_arr)
        elif method == 'skipped':
            r, pval, outliers = skipped(x_arr, y_arr, method='spearman')
        else:
            raise ValueError('Method not recognized.')
    elif _continuous_bool(x, y):
        # sort them into order, it matters
        r, pval = pointbiserialr(x_arr, y_arr.astype(np.uint8))
        # override method
        method = "biserial"
    elif _bool_continuous(x, y):
        # sort them into order, it matters
        r, pval = pointbiserialr(x_arr.astype(np.uint8), y_arr)
        # override method
        method = "biserial"
    elif _boolbool(x, y):
        # use spearman
        r, pval = spearmanr(x_arr.astype(np.uint8), y_arr.astype(np.uint8))
        method = "spearman"
    else:
        raise TypeError(
            "columns '{}':{} to '{}':{} combination not accepted for `bicorr`.".format(x.name, x.dtype, y.name,
                                                                                       y.dtype))
    assert not np.isnan(r), 'Correlation returned NaN. Check your data.'

    if verbose > 0:
        print("correlating {}:{}".format(x.name, y.name))

    # Compute r2 and adj_r2
    r2 = r ** 2
    adj_r2 = 1 - (((1 - r2) * (nx - 1)) / (nx - 3))

    # Compute the parametric 95% confidence interval and power
    if r2 < 1:
        ci = compute_esci(stat=r, nx=nx, ny=nx, eftype='r')
        pr = round(power_corr(r=r, n=nx, power=None, alpha=0.05, tail=tail), 3)
    else:
        ci = [1., 1.]
        pr = np.inf

    # Create dictionary
    sd_d = {'x': x.name, 'y': y.name, 'n': nx,
            'r': round(r, 3),
            'r2': round(r2, 3),
            'adj_r2': round(adj_r2, 3),
            'CI95_lower': ci[0], 'CI95_upper': ci[1],
            'p_val': pval if tail == 'two-sided' else .5 * pval,
            'power': pr,
            'outliers': sum(outliers) if method in ('shepherd', 'skipped') else np.nan}

    # Convert to DataFrame
    _stm = pd.DataFrame.from_records(sd_d, index=[method])
    return _stm


def _partial_bicorr_inner(data, x, y, covar, tail='two-sided', method='spearman', verbose=0):
    """Internal method for partial bi correlation here."""
    # all columns select
    if verbose > 0:
        print("partial {}:{}\\{}".format(x, y, covar))
    col = union(x, y, covar)
    """ Calculate linear models here to get residuals for x, y to correlate together. """
    # Drop rows with NaN
    _data = data[col].dropna()
    # use LM to generate predictions
    px, r_x = lm(_data[covar], _data[x])
    py, r_y = lm(_data[covar], _data[y])
    # wrap residuals as series
    # if one is a boolean operation, we must preserve structure
    res_x = pd.Series(r_x, name=x)
    res_y = pd.Series(r_y, name=y)
    """ Perform bivariate correlate as normal. """
    # calculate bicorrelation on residuals
    return _bicorr_inner(res_x, res_y, method=method, tail=tail, verbose=0)


""" Public bivariate methods. """


def bicorr(x: pd.Series,
           y: pd.Series,
           tail: str = 'two-sided',
           method: str = 'spearman') -> pd.DataFrame:
    """(Robust) correlation between two variables.

    Adapted from the `pingouin` library, made by Raphael Vallat.

    .. [1] https://github.com/raphaelvallat/pingouin/blob/master/pingouin/correlation.py

    Parameters
    ----------
    x, y : pd.Series
        First and second set of observations. x and y must be independent.
    tail : string
        Specify whether to return 'one-sided' or 'two-sided' p-value.
    method : string
        Specify which method to use for the computation of the correlation
        coefficient. Available methods are ::
        'pearson' : Pearson product-moment correlation
        'spearman' : Spearman rank-order correlation
        'kendall' : Kendall’s tau (ordinal data)
        'biserial' : Biserial correlation (continuous and boolean data)
        'percbend' : percentage bend correlation (robust)
        'shepherd' : Shepherd's pi correlation (robust Spearman)
        'skipped' : skipped correlation (robust Spearman, requires sklearn)

    Returns
    -------
    stats : pandas DataFrame
        Test summary ::
        'n' : Sample size (after NaN removal)
        'outliers' : number of outliers (only for 'shepherd' or 'skipped')
        'r' : Correlation coefficient
        'CI95' : 95% parametric confidence intervals
        'r2' : R-squared
        'adj_r2' : Adjusted R-squared
        'p-val' : one or two tailed p-value
        'power' : achieved power of the test (= 1 - type II error).
    Notes
    -----
    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Correlations of -1 or +1 imply
    an exact linear relationship.
    The Spearman correlation is a nonparametric measure of the monotonicity of
    the relationship between two datasets. Unlike the Pearson correlation,
    the Spearman correlation does not assume that both datasets are normally
    distributed. Correlations of -1 or +1 imply an exact monotonic
    relationship.
    Kendall’s tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, values close to -1 indicate
    strong disagreement.
    The percentage bend correlation [1]_ is a robust method that
    protects against univariate outliers.
    The Shepherd's pi [2]_ and skipped [3]_, [4]_ correlations are both robust
    methods that returns the Spearman's rho after bivariate outliers removal.
    Note that the skipped correlation requires that the scikit-learn
    package is installed (for computing the minimum covariance determinant).
    Please note that rows with NaN are automatically removed.
    References
    ----------
    .. [1] Wilcox, R.R., 1994. The percentage bend correlation coefficient.
       Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395
    .. [2] Schwarzkopf, D.S., De Haas, B., Rees, G., 2012. Better ways to
       improve standards in brain-behavior correlation analysis. Front.
       Hum. Neurosci. 6, 200. https://doi.org/10.3389/fnhum.2012.00200
    .. [3] Rousselet, G.A., Pernet, C.R., 2012. Improving standards in
       brain-behavior correlation analyses. Front. Hum. Neurosci. 6, 119.
       https://doi.org/10.3389/fnhum.2012.00119
    .. [4] Pernet, C.R., Wilcox, R., Rousselet, G.A., 2012. Robust correlation
       analyses: false positive and power validation using a new open
       source matlab toolbox. Front. Psychol. 3, 606.
       https://doi.org/10.3389/fpsyg.2012.00606
    """
    # perform all checks in the public method.. rather than repeating them internally.
    # check type
    instance_check((x, y), pd.Series)
    belongs(tail, ("one-sided", "two-sided"))
    belongs(method, ('pearson', 'spearman', 'kendall', 'biserial', 'percbend', 'shepherd', 'skipped'))
    # Check size
    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y must have the same length.')

    return _bicorr_inner(x, y, tail, method)


def partial_bicorr(data: pd.DataFrame,
                   x: str,
                   y: str,
                   covar: Union[str, List[str]],
                   tail: str = 'two-sided',
                   method: str = 'spearman') -> pd.DataFrame:
    """Partial and semi-partial correlation.

    Adapted from the `pingouin` library, made by Raphael Vallat.

    .. [1] https://github.com/raphaelvallat/pingouin/blob/master/pingouin/correlation.py

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset including covariates.
    x, y : str, list of str
        x and y. Must be names of columns in ``data``.
    covar : list of str
        Covariate(s). Column names of the covariates. covar must be made of continuous columns.
        If x, y are not continuous, will perform logistic regression to generate residuals.
    tail : string
        Specify whether to return the 'one-sided' or 'two-sided' p-value.
    method : string
        Specify which method to use for the computation of the correlation
        coefficient. Available methods are ::
        'pearson' : Pearson product-moment correlation
        'spearman' : Spearman rank-order correlation
        'biserial' : Biserial correlation (continuous and boolean data)
        'kendall' : Kendall’s tau (ordinal data)
        'percbend' : percentage bend correlation (robust)
        'shepherd' : Shepherd's pi correlation (robust Spearman)
        'skipped' : skipped correlation (robust Spearman, requires sklearn)
    Returns
    -------
    stats : pandas DataFrame
        Test summary ::
        'n' : Sample size (after NaN removal)
        'outliers' : number of outliers (only for 'shepherd' or 'skipped')
        'r' : Correlation coefficient
        'CI95' : 95% parametric confidence intervals
        'r2' : R-squared
        'adj_r2' : Adjusted R-squared
        'p-val' : one or two tailed p-value
        'BF10' : Bayes Factor of the alternative hypothesis (Pearson only)
        'power' : achieved power of the test (= 1 - type II error).
    Notes
    -----
    From [4]_:
    “With *partial correlation*, we find the correlation between :math:`x`
    and :math:`y` holding :math:`C` constant for both :math:`x` and
    :math:`y`. Sometimes, however, we want to hold :math:`C` constant for
    just :math:`x` or just :math:`y`. In that case, we compute a
    *semi-partial correlation*. A partial correlation is computed between
    two residuals. A semi-partial correlation is computed between one
    residual and another raw (or unresidualized) variable.”
    Note that if you are not interested in calculating the statistics and
    p-values but only the partial correlation matrix, a (faster)
    alternative is to use the :py:func:`pingouin.pcorr` method (see example 4).
    Rows with missing values are automatically removed from data. Results have
    been tested against the `ppcor` R package.
    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Partial_correlation
    .. [3] https://cran.r-project.org/web/packages/ppcor/index.html
    .. [4] https://gist.github.com/fabianp/9396204419c7b638d38f
    .. [5] http://faculty.cas.usf.edu/mbrannick/regression/Partial.html
    """
    # perform all checks in the public method..
    instance_check(data, pd.DataFrame)
    instance_check((x, y), str)
    instance_check(covar, (str, list, tuple, pd.Index))
    belongs(tail, ('one-sided', 'two-sided'))
    belongs(method, ('pearson', 'spearman', 'kendall', 'biserial', 'percbend', 'shepherd', 'skipped'))

    # perform a check to make sure every column in `covar` is continuous.
    if not is_dataframe_float(data[covar]):
        raise TypeError("`covar` variables in `partial_bicorr` all must be of type `float`/continuous.")

    return _partial_bicorr_inner(data, x, y, covar, tail, method)


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
        # return _corr_matrix_singular(df, covar=covar, method=method, cartesian_Z=cartesian_covar, verbose=verbose)
    elif isinstance(x, (list, tuple, pd.Index)) and y is None:
        # use a subset of x, in union with covar
        comb = it.combinations_with_replacement(x, 2)
        # return _corr_matrix_singular(df[union(x, covar)], covar=covar, method=method, cartesian_Z=cartesian_covar, verbose=verbose)
    elif isinstance(x, (list, tuple, pd.Index)) and isinstance(y, str):
        # list of x, y str -> matrix-vector cartesian product
        comb = it.product(x, [y])
        # return _corr_matrix_vector(df, x, y, covar=covar, method=method, cartesian_Z=cartesian_covar, verbose=verbose)
    elif isinstance(y, (list, tuple, pd.Index)) and isinstance(x, str):
        # list of y, x str -> matrix-vector cartesian product
        comb = it.product(y, [x])
        # return _corr_matrix_vector(df, y, x, covar=covar, method=method, cartesian_Z=cartesian_covar, verbose=verbose)
    elif isinstance(x, (list, tuple, pd.Index)) and isinstance(y, (list, tuple, pd.Index)):
        # list of x, y -> cartesian product of x: y terms
        comb = it.product(x, y)
        # return _corr_two_matrix_diff(df, x, y, covar=covar, method=method, cartesian_Z=cartesian_covar, verbose=verbose)
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
