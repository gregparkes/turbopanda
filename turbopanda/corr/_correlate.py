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

import itertools as it
# imports
from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd

# locals
from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda._deprecator import deprecated
from turbopanda.utils import instance_check, union, remove_na, \
    is_column_float, belongs, difference, is_column_boolean

# user define dataset type
DataSetType = Union[pd.Series, pd.DataFrame, MetaPanda]


__all__ = ('correlate', 'bicorr', 'partial_bicorr', 'row_to_matrix')


def _both_continuous(x, y):
    return is_column_float(x) and is_column_float(y)


def _continuous_bool(x, y):
    return (is_column_float(x) and is_column_boolean(y)) or (is_column_float(y) and is_column_boolean(x))


def _get_continuous_bool_order(x, y):
    return (x, y) if (is_column_float(x) and is_column_boolean(y)) else (y, x)


def _boolbool(x, y):
    return is_column_boolean(x) and is_column_boolean(y)


def _row_to_matrix(rows: pd.DataFrame, y_column="r") -> pd.DataFrame:
    """Takes the verbose row output and converts to lighter matrix format."""
    square = rows.pivot_table(index="x", columns="y", values=y_column)
    # fillna
    square.fillna(0.0, inplace=True)
    # ready for transpose
    square += square.T
    # eliminate 1 from diag
    square.values[np.diag_indices_from(square.values)] -= 1.
    return square


def _compute_residuals(C, x, y, Z):
    """Given Z and x and y of continuous float, produce residuals e_x, e_y

    TODO: Implement residual calculations for continuous and dichotonomous for partial correlation
         For continuous, use least-squares residuals. For bool-like, use logistic regression?

    """
    return NotImplemented


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
    from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr
    from ._corr_metrics import percbend, shepherd, skipped
    from ._stats_extra import compute_esci, power_corr

    # check type
    if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
        raise TypeError("x:{} or y:{} is not of type [pd.Series]".format(type(x), type(y)))
    # convert to numpy
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    # Check size
    if x_arr.size != y_arr.size:
        raise ValueError('x and y must have the same length.')

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
        x_arr, y_arr = _get_continuous_bool_order(x_arr, y_arr)
        r, pval = pointbiserialr(x_arr, y_arr.astype(np.uint8))
        # override method
        method = "biserial"
    elif _boolbool(x, y):
        # use spearman
        r, pval = spearmanr(x_arr.astype(np.uint8), y_arr.astype(np.uint8))
        method = "spearman"
    else:
        raise TypeError(
            "columns '{}':{} to '{}':{} combination not accepted for `_bicorr`.".format(x.name, x.dtype, y.name,
                                                                                        y.dtype))
    assert not np.isnan(r), 'Correlation returned NaN. Check your data.'

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
    stats = {'x': x.name, 'y': y.name, 'n': nx, 'r': round(r, 3), 'r2': round(r2, 3), 'adj_r2': round(adj_r2, 3),
             'CI95_lower': ci[0], 'CI95_upper': ci[1], 'p-val': pval if tail == 'two-sided' else .5 * pval, 'power': pr,
             'outliers': sum(outliers) if method in ('shepherd', 'skipped') else np.nan}

    # Convert to DataFrame
    stats = pd.DataFrame.from_records(stats, index=[method])
    return stats


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
    instance_check(data, pd.DataFrame)
    instance_check(x, str)
    instance_check(y, str)
    instance_check(covar, (str, list, tuple, pd.Index))
    # Check that columns exist
    col = union(x, y, covar)
    # Drop rows with NaN
    _data = data[col].dropna()
    assert _data.shape[0] > 2, 'Data must have at least 3 non-NAN samples.'

    # Standardize (= no need for an intercept in least-square regression)
    C = (_data[col] - _data[col].mean(axis=0)) / _data[col].std(axis=0)
    # PARTIAL CORRELATION - HANDLING SCENARIOS
    cvar = np.atleast_2d(C[covar].values)
    beta_x = np.linalg.lstsq(cvar, C[x].values, rcond=None)[0]
    beta_y = np.linalg.lstsq(cvar, C[y].values, rcond=None)[0]
    # determine residuals and wrap pandas
    res_x = pd.Series(C[x].values - np.dot(cvar, beta_x), name=x)
    res_y = pd.Series(C[y].values - np.dot(cvar, beta_y), name=y)
    # calculate bicorrelation on residuals
    return bicorr(res_x, res_y, method=method, tail=tail)


def _corr_two_matrix_diff(data: pd.DataFrame,
                          x: List[str],
                          y: List[str],
                          covar: Union[str, List[str]] = None,
                          method: str = 'spearman') -> pd.DataFrame:
    """
    Computes the correlation between two matrices X and Y of different columns lengths.

    Essentially computes multiple iterations of corr_matrix_vector.
    """
    instance_check(data, pd.DataFrame)
    instance_check(x, (list, tuple, pd.Index))
    instance_check(y, (list, tuple, pd.Index))
    instance_check(covar, (type(None), str, list, tuple, pd.Index))

    # create combinations
    comb = list(it.product(x, y))
    # iterate and perform two_variable as before
    if covar is None:
        result_k = [bicorr(data[xc], data[yc], method=method) for (xc, yc) in comb]
    else:
        result_k = [partial_bicorr(data, xc, yc, covar, method=method) for (xc, yc) in comb]

    result = pd.concat(result_k, axis=0, sort=False).reset_index().rename(columns={"index": "method"})
    return result


def _corr_matrix_singular(data: pd.DataFrame,
                          covar: Union[str, List[str]] = None,
                          method: str = "spearman") -> pd.DataFrame:
    """Computes the correlation matrix on pairwise columns."""
    # assign zeros/empty
    instance_check(data, pd.DataFrame)
    instance_check(covar, (type(None), str, list, tuple, pd.Index))

    ac = []
    # iterate over i,j pairs
    if covar is None:
        for i in range(data.shape[1]):
            for j in range(i, data.shape[1]):
                ac.append(
                    bicorr(data.iloc[:, i], data.iloc[:, j],
                           method=method)
                )
    else:
        # computes the symmetric difference
        _x = difference(data.columns, covar)
        # iterate and append
        for i in range(_x.shape[1]):
            for j in range(i, _x.shape[1]):
                ac.append(
                    partial_bicorr(data, _x[i], _x[j], covar, method=method)
                )
    results = pd.concat(ac, axis=0, sort=False).reset_index().rename(columns={"index": "method"})
    return results


def _corr_matrix_vector(data: pd.DataFrame,
                        x: List[str],
                        y: str,
                        covar: Union[str, List[str]] = None,
                        method: str = "spearman") -> pd.DataFrame:
    instance_check(data, pd.DataFrame)
    instance_check(x, (list, tuple, pd.Index))
    instance_check(y, str)
    instance_check(covar, (type(None), str, list, tuple, pd.Index))

    # join together the list/strings of column names
    cols = union(x, y, covar)
    # extract data subset, no dropping yet though.
    _data = data[cols]

    # calculate bicorr or partial bicorr based on presense of covar list
    if covar is None:
        _x = _data[x]
        _y = _data[y]
        result_k = [bicorr(_x.iloc[:, i], _y, method=method) for i in range(_x.shape[1])]
    else:
        result_k = [partial_bicorr(_data, _xn, y, covar, method=method) for _xn in x]

    # concatenate, add method and return
    result = pd.concat(result_k, axis=0, sort=False).reset_index().rename(columns={"index": "method"})
    return result


"""##################### PUBLIC FUNCTIONS ####################################################### """


@deprecated("0.2.4", '0.2.6', instead="")
def correlate_mat(data, covar=None, method="spearman") -> pd.DataFrame:
    """Correlates data matrix into row set. No additional parameters.

    Parameters
    ----------
    data : pd.DataFrame / MetaPanda
        The full dataset.
    covar : (str, list, tuple, pd.Index), optional
        set of covariate(s). Covariates are needed to compute partial correlations.
            If None, uses standard correlation.
    method : str, optional
        Method to correlate with. Choose from:
            'pearson' : Pearson product-moment correlation
            'spearman' : Spearman rank-order correlation
            'kendall' : Kendall’s tau (ordinal data)
            'biserial' : Biserial correlation (continuous and boolean data only)
            'percbend' : percentage bend correlation (robust)
            'shepherd' : Shepherd's pi correlation (robust Spearman)
            'skipped' : skipped correlation (robust Spearman, requires sklearn)

    Returns
    -------
    R : pd.DataFrame
        correlation rows (based on pingouin structure)
    """
    instance_check(data, (pd.DataFrame, MetaPanda))

    args = dict(covar=covar, method=method)
    return _corr_matrix_singular(data, **args) if isinstance(data, pd.DataFrame) \
        else _corr_matrix_singular(data.df_, **args)


def correlate(data: Union[pd.DataFrame, MetaPanda],
              x: Optional[SelectorType] = None,
              y: Optional[SelectorType] = None,
              covar: Optional[SelectorType] = None,
              method: str = "spearman") -> pd.DataFrame:
    """Correlates X and Y together to generate a list of correlations.

    If X/Y are MetaPandas, returns a MetaPanda object, else returns pandas.DataFrame

    Parameters
    ---------
    data : pd.DataFrame / MetaPanda
        The full dataset.
    x : (str, list, tuple, pd.Index), optional
        Subset of input(s) for column names.
            if None, uses the full dataset. Y must be None in this case also.
    y : (str, list, tuple, pd.Index)
        Subset of output(s) for column names.
            if None, uses the full dataset (from optional `x` subset)
    covar : (str, list, tuple, pd.Index), optional
        set of covariate(s). Covariates are needed to compute partial correlations.
            If None, uses standard correlation.
    method : str, optional
        Method to correlate with. Choose from:
            'pearson' : Pearson product-moment correlation
            'spearman' : Spearman rank-order correlation
            'kendall' : Kendall’s tau (ordinal data)
            'biserial' : Biserial correlation (continuous and boolean data only)
            'percbend' : percentage bend correlation (robust)
            'shepherd' : Shepherd's pi correlation (robust Spearman)
            'skipped' : skipped correlation (robust Spearman, requires sklearn)
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
    instance_check(x, (type(None), str, list, tuple, pd.Index))
    instance_check(y, (type(None), str, list, tuple, pd.Index))
    instance_check(covar, (type(None), str, list, tuple, pd.Index))
    belongs(method, ('pearson', 'spearman', 'kendall', 'biserial', 'percbend', 'shepherd', 'skipped'))
    # downcast to dataframe option
    df = data.df_ if isinstance(data, MetaPanda) else data
    # downcast if list/tuple/pd.index is of length 1
    x = x[0] if (isinstance(x, (tuple, list, pd.Index)) and len(x) == 1) else x
    y = y[0] if (isinstance(y, (tuple, list, pd.Index)) and len(y) == 1) else y

    # convert using `view` if we have string instances.
    if isinstance(x, str) and isinstance(data, MetaPanda):
        x = data.view(x)
    if isinstance(y, str) and isinstance(data, MetaPanda):
        y = data.view(y)

    if x is None and y is None:
        return _corr_matrix_singular(df, covar=covar, method=method)
    elif isinstance(x, (list, tuple, pd.Index)) and y is None:
        cols = union(x, covar)
        return _corr_matrix_singular(df[cols], covar=covar, method=method)
    elif isinstance(x, (list, tuple, pd.Index)) and isinstance(y, str):
        return _corr_matrix_vector(df, x, y, covar=covar, method=method)
    elif isinstance(y, (list, tuple, pd.Index)) and isinstance(x, str):
        return _corr_matrix_vector(df, y, x, covar=covar, method=method)
    elif isinstance(x, (list, tuple, pd.Index)) and isinstance(y, (list, tuple, pd.Index)):
        return _corr_two_matrix_diff(df, x, y, covar=covar, method=method)
    else:
        raise ValueError("XC: {}; YC: {}; COVA: {} combination unknown.".format(x, y, covar))


def row_to_matrix(rows: pd.DataFrame, piv_value='r'):
    """Converts a row-output from `correlate` into matrix form.

    Parameters
    ----------
    rows : pd.DataFrame
        The output from `correlate`
    piv_value : str, optional
        Which parameter to pivot on, by default is `r`, the coefficient.

    Returns
    -------
    m : pd.DataFrame (p, p)
        The correlation matrix
    """
    return _row_to_matrix(rows, y_column=piv_value)


#########################################################################################################
# EXTRACTS TAKEN FROM PINGOUIN LIBRARY..
#########################################################################################################


@deprecated("0.2.4", "0.2.6", instead="`correlate`", reason="partial correlations are now factored into correlate() "
                                                            "with `covar` argument")
def pcm(df: Union[pd.DataFrame, MetaPanda]) -> pd.DataFrame:
    """Partial correlation matrix.

    Adapted from the `pingouin` library, made by Raphael Vallat.

    .. [1] https://github.com/raphaelvallat/pingouin/blob/master/pingouin/correlation.py

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe matrix to calculate on.

    Returns
    ----------
    pcormat : :py:class:`pandas.DataFrame`
        Partial correlation matrix.
    Notes
    -----
    This function calculates the pairwise partial correlations for each pair of
    variables in a :py:class:`pandas.DataFrame` given all the others. It has
    the same behavior as the pcor function in the `ppcor` R package.
    Note that this function only returns the raw Pearson correlation
    coefficient. If you want to calculate the test statistic and p-values, or
    use more robust estimates of the correlation coefficient, please refer to
    the :py:func:`pingouin.pairwise_corr` or :py:func:`pingouin.partial_corr`
    functions. The :py:func:`pingouin.pcorr` function uses the inverse of
    the variance-covariance matrix to calculate the partial correlation matrix
    and is therefore much faster than the two latter functions which are based
    on the residuals.

    References
    ----------
    .. [2] https://cran.r-project.org/web/packages/ppcor/index.html
    Examples
    --------
    >>> import turbopanda as turb
    >>> data = turb.read('example.json')
    >>> turb.pcm(data)
                 X         M         Y      Mbin      Ybin
    X     1.000000  0.392251  0.059771 -0.014405 -0.149210
    M     0.392251  1.000000  0.545618 -0.015622 -0.094309
    Y     0.059771  0.545618  1.000000 -0.007009  0.161334
    Mbin -0.014405 -0.015622 -0.007009  1.000000 -0.076614
    Ybin -0.149210 -0.094309  0.161334 -0.076614  1.000000
    """
    df = df.df_ if isinstance(df, MetaPanda) else df
    V = df.cov()  # Covariance matrix
    Vi = np.linalg.pinv(V)  # Inverse covariance matrix
    D = np.diag(np.sqrt(1 / np.diag(Vi)))
    pcor = -1 * (D @ Vi @ D)  # Partial correlation matrix
    pcor[np.diag_indices_from(pcor)] = 1
    return pd.DataFrame(pcor, index=V.index, columns=V.columns)
