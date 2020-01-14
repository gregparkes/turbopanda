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
from typing import Union, List, Tuple, Optional, Dict

import warnings
import numpy as np
import pandas as pd

from .custypes import ArrayLike
# locals
from .metapanda import MetaPanda
from .utils import c_float, intcat, instance_check, dictzip, join, remove_na, is_column_float, is_column_intbool

# user define dataset type
DataSetType = Union[pd.Series, pd.DataFrame, MetaPanda]

__all__ = ['correlate', 'partial_correlate', 'pcm']


def _both_continuous(x, y):
    return is_column_float(x) and is_column_float(y)


def _continuous_bool(x, y):
    return (is_column_float(x) and is_column_intbool(y)) or (is_column_float(y) and is_column_intbool(x))


def _get_continuous_bool(x, y):
    """Returns first elem as continuous, second as the bool"""
    if is_column_float(x) and is_column_intbool(y):
        return x, y
    else:
        return y, x


def _boolbool(x, y):
    return is_column_intbool(x) and is_column_intbool(y)


def _wrap_corr_metapanda(df_corr, pdm):
    mpf = MetaPanda(df_corr)
    # copy over metadata - dropping columns that aren't present in df_corr
    mpf._meta = pdm.meta_.loc[df_corr.columns]
    # copy over selector
    mpf._select = pdm.selectors_
    # copy over name
    mpf.name_ = pdm.name_
    return mpf


def _row_to_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Takes the verbose row output and converts to lighter matrix format."""
    square = rows.pivot("column1", "column2", "r")
    # fillna
    square.fillna(0.0, inplace=True)
    # ready for transpose
    square += square.T
    # eliminate 1 from diag
    square.values[np.diag_indices_from(square.values)] -= 1.
    return square


def _bicorr(x: ArrayLike,
            y: ArrayLike,
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
        x_arr, y_arr = _get_continuous_bool(x_arr, y_arr)
        r, pval = pointbiserialr(x_arr, y_arr)
        # override method
        method = "biserial"
    elif _boolbool(x, y):
        # use spearman
        r, pval = spearmanr(x_arr, y_arr)
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
    stats = {
        'x': x.name,
        'y': y.name,
        'n': nx,
        'r': round(r, 3),
        'r2': round(r2, 3),
        'adj_r2': round(adj_r2, 3),
        'CI95_lower': ci[0],
        'CI95_upper': ci[1],
        'p-val': pval if tail == 'two-sided' else .5 * pval,
        'power': pr
    }

    if method in ('shepherd', 'skipped'):
        stats['outliers'] = sum(outliers)

    # Convert to DataFrame
    stats = pd.DataFrame.from_records(stats, index=[method])

    # Define order
    # col_keep = ['x', 'y', 'n', 'outliers', 'r', 'CI95%', 'r2', 'adj_r2', 'p-val', 'power']
    # col_order = [k for k in col_keep if k in stats.keys().tolist()]
    # return stats[col_order]
    return stats


def _partial_bicorr(data: pd.DataFrame,
                    x: Union[str, List[str]],
                    y: Union[str, List[str]],
                    covar: List[str],
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
        Covariate(s). Column names of the covariates.
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
    # Check arguments
    assert isinstance(data, pd.DataFrame), 'data must be a pandas DataFrame.'
    assert data.shape[0] > 2, 'Data must have at least 3 samples.'
    assert isinstance(x, (str, tuple)), 'x must be a string.'
    assert isinstance(y, (str, tuple)), 'y must be a string.'
    assert isinstance(covar, (str, list, type(None)))
    assert isinstance(x_covar, (str, list, type(None)))
    assert isinstance(y_covar, (str, list, type(None)))
    if covar is not None and (x_covar is not None or y_covar is not None):
        raise ValueError('Cannot specify both covar and {x,y}_covar.')
    # Check that columns exist
    col = join(x, y, covar)

    # Drop rows with NaN
    data = data[col].dropna()
    assert data.shape[0] > 2, 'Data must have at least 3 non-NAN samples.'

    # Standardize (= no need for an intercept in least-square regression)
    C = (data[col] - data[col].mean(axis=0)) / data[col].std(axis=0)
    # PARTIAL CORRELATION
    cvar = np.atleast_2d(C[covar].values)
    beta_x = np.linalg.lstsq(cvar, C[x].values, rcond=None)[0]
    beta_y = np.linalg.lstsq(cvar, C[y].values, rcond=None)[0]
    # determine residuals and wrap pandas
    res_x = pd.Series(C[x].values - np.dot(cvar, beta_x), name=x)
    res_y = pd.Series(C[y].values - np.dot(cvar, beta_y), name=y)
    # calculate bicorrelation on residuals
    return _bicorr(res_x, res_y, method=method, tail=tail)


def _corr_two_matrix_same(x: pd.DataFrame,
                          y: pd.DataFrame,
                          method: str = "spearman") -> pd.DataFrame:
    """
    Computes the correlation between two matrices X and Y of same size.

    No debug of this matrix, since all the names are the same.
    """
    cor = np.zeros((x.shape[1], 8))
    if x.shape != y.shape:
        raise ValueError("X.shape {} does not match Y.shape {}".format(x.shape, y.shape))

    return pd.concat(
        [_bicorr(x.iloc[:, i], y.iloc[:, i], method=method) for i in range(x.shape[1])],
        axis=0, sort=False
    ).reset_index().rename(columns={"index": "method"})


def _corr_two_matrix_diff(x: pd.DataFrame,
                          y: pd.DataFrame,
                          method: str = 'spearman') -> pd.DataFrame:
    """
    Computes the correlation between two matrices X and Y of different columns lengths.

    Essentially computes multiple iterations of corr_matrix_vector.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("X.rows: {} does not match Y.rows: {}".format(x.shape[0], y.shape[1]))
    # create combinations
    comb = list(it.product(x.columns.tolist(), y.columns.tolist()))
    # iterate and perform two_variable as before
    data = pd.concat([
        _bicorr(x[xcols], y[ycols], method=method) for (xcols, ycols) in comb
    ], axis=0, sort=False).reset_index().rename(columns={"index": "method"})
    return data


def _corr_matrix_singular(x: pd.DataFrame,
                          method: str = "spearman") -> pd.DataFrame:
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
    if sel_keep.sum() <= 1:
        raise ValueError("there are no non-object-like columns in X")
    elif sel_keep.sum() == 2:
        cols = x.columns[sel_keep]
        return _bicorr(x[cols[0]], x[cols[1]], method=method)
    else:
        contin_x = x.loc[:, sel_keep]
        # assign zeros/empty
        ac = []
        # iterate over i,j pairs
        for i in range(contin_x.shape[1]):
            for j in range(i, contin_x.shape[1]):
                ac.append(
                    _bicorr(contin_x.iloc[:, i], contin_x.iloc[:, j],
                            method=method)
                )
        data = pd.concat(ac, axis=0, sort=False).reset_index().rename(columns={"index": "method"})
        return data


def _corr_matrix_vector(x: pd.DataFrame,
                        y: pd.Series,
                        method: str = "spearman") -> pd.DataFrame:
    if x.shape[0] != y.shape[0]:
        raise ValueError("X.shape {} does not match Y.shape {}".format(x.shape, y.shape))

    data = pd.concat(
        [_bicorr(x.iloc[:, i], y, method=method) for i in range(x.shape[1])],
        axis=0, sort=False).reset_index().rename(columns={"index": "method"})
    return data


"""##################### PUBLIC FUNCTIONS ####################################################### """


def correlate(x: Union[str, List, Tuple, pd.Index, DataSetType],
              y: Optional[Union[str, List, Tuple, pd.Index, DataSetType]] = None,
              data: Optional[Union[pd.DataFrame, MetaPanda]] = None,
              method: str = "spearman",
              style: str = "rows") -> Union[pd.DataFrame, MetaPanda]:
    """Correlates X and Y together to generate a correlation matrix.

    If X/Y are MetaPandas, returns a MetaPanda object, else returns pandas.DataFrame

    ..warning:: Arguments will likely be rearranged/renamed in future updates 0.2.1 onwards.

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
        Method to correlate with. Choose from:
            'pearson' : Pearson product-moment correlation
            'spearman' : Spearman rank-order correlation
            'kendall' : Kendall’s tau (ordinal data)
            'biserial' : Biserial correlation (continuous and boolean data)
            'percbend' : percentage bend correlation (robust)
            'shepherd' : Shepherd's pi correlation (robust Spearman)
            'skipped' : skipped correlation (robust Spearman, requires sklearn)
    style : str, optional
        Choose from {'matrix', 'rows'}
        If 'matrix': returns a pandas.DataFrame square matrix
        If 'rows': returns row-wise (x, y) correlation on each row of pandas.DataFrame (contains more information)
            Note this type only works if X and Y are different, and both are not of type {pd.Series}

    Returns
    -------
    R : dict/pd.DataFrame/turb.MetaPanda
        correlation matrix/rows
    """
    warnings.warn(
        "in `correlate` from version 0.2.1 onwards, there will likely be changes to parameter order and name.",
        FutureWarning
    )

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
        NX = data.df_[X_c].squeeze() if isinstance(data, MetaPanda) else data[X_c]
        if y is None:
            NY = None
        else:
            NY = data.df_[Y_c].squeeze() if isinstance(data, MetaPanda) else data[Y_c]

    """ Handle different use cases....
        1. Y is None, and we have a DataFrame
        2. X and Y are series
        3. X is DataFrame and Y is seres
        4. X and Y are DataFrame of same size
    """
    # if no Y. do singular matrix.
    if NY is None and isinstance(NX, pd.DataFrame):
        mat = _corr_matrix_singular(NX, method=method)
        if isinstance(x, MetaPanda):
            return _wrap_corr_metapanda(mat, x)
        else:
            return mat
    # two series.
    if isinstance(NX, pd.Series) and isinstance(NY, pd.Series):
        return _bicorr(NX, NY, method=method)
    # one matrix, one vector
    elif isinstance(NX, pd.DataFrame) and isinstance(NY, pd.Series):
        return _corr_matrix_vector(NX, NY, method=method)
    # one vector, one matrix
    elif isinstance(NX, pd.Series) and isinstance(NY, pd.DataFrame):
        # swap them over
        return _corr_matrix_vector(NY, NX, method=method)
    # two matrices of same shape
    elif isinstance(NX, pd.DataFrame) and isinstance(NY, pd.DataFrame):
        if NX.shape[1] == NY.shape[1]:
            return _corr_two_matrix_same(NX, NY, method=method)
        else:
            return _corr_two_matrix_diff(NX, NY, method=method)
    else:
        raise TypeError("X of type [{}], Y of type [{}], cannot compare".format(type(NX), type(NY)))


#########################################################################################################
# EXTRACTS TAKEN FROM PINGOUIN LIBRARY..
#########################################################################################################


def partial_correlate(data=None, x=None, y=None, covar=None, x_covar=None,
                      y_covar=None, tail='two-sided', method='pearson'):
    """Partial and semi-partial correlation.

    Adapted from the `pingouin` library, made by Raphael Vallat.

    .. [1] https://github.com/raphaelvallat/pingouin/blob/master/pingouin/correlation.py

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe. Note that this function can also directly be used as a
        :py:class:`pandas.DataFrame` method, in which case this argument is
        no longer needed.
    x, y : string
        x and y. Must be names of columns in ``data``.
    covar : string or list
        Covariate(s). Must be a names of columns in ``data``. Use a list if
        there are two or more covariates.
    x_covar : string or list
        Covariate(s) for the ``x`` variable. This is used to compute
        semi-partial correlation (i.e. the effect of ``x_covar`` is removed
        from ``x`` but not from ``y``). Note that you cannot specify both
        ``covar`` and ``x_covar``.
    y_covar : string or list
        Covariate(s) for the ``y`` variable. This is used to compute
        semi-partial correlation (i.e. the effect of ``y_covar`` is removed
        from ``y`` but not from ``x``). Note that you cannot specify both
        ``covar`` and ``y_covar``.
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
    # Check arguments
    assert isinstance(data, pd.DataFrame), 'data must be a pandas DataFrame.'
    assert data.shape[0] > 2, 'Data must have at least 3 samples.'
    assert isinstance(x, (str, tuple)), 'x must be a string.'
    assert isinstance(y, (str, tuple)), 'y must be a string.'
    assert isinstance(covar, (str, list, type(None)))
    assert isinstance(x_covar, (str, list, type(None)))
    assert isinstance(y_covar, (str, list, type(None)))
    if covar is not None and (x_covar is not None or y_covar is not None):
        raise ValueError('Cannot specify both covar and {x,y}_covar.')
    # Check that columns exist
    col = join(x, y, covar, x_covar, y_covar)
    if isinstance(covar, str):
        covar = [covar]
    if isinstance(x_covar, str):
        x_covar = [x_covar]
    if isinstance(y_covar, str):
        y_covar = [y_covar]
    assert all([c in data for c in col]), 'columns are not in dataframe.'
    # Check that columns are numeric
    assert all([data[c].dtype.kind in 'bfi' for c in col])

    # Drop rows with NaN
    data = data[col].dropna()
    assert data.shape[0] > 2, 'Data must have at least 3 non-NAN samples.'

    # Standardize (= no need for an intercept in least-square regression)
    C = (data[col] - data[col].mean(axis=0)) / data[col].std(axis=0)

    if covar is not None:
        # PARTIAL CORRELATION
        cvar = np.atleast_2d(C[covar].values)
        beta_x = np.linalg.lstsq(cvar, C[x].values, rcond=None)[0]
        beta_y = np.linalg.lstsq(cvar, C[y].values, rcond=None)[0]
        res_x = C[x].values - np.dot(cvar, beta_x)
        res_y = C[y].values - np.dot(cvar, beta_y)
    else:
        # SEMI-PARTIAL CORRELATION
        # Initialize "fake" residuals
        res_x, res_y = data[x].values, data[y].values
        if x_covar is not None:
            cvar = np.atleast_2d(C[x_covar].values)
            beta_x = np.linalg.lstsq(cvar, C[x].values, rcond=None)[0]
            res_x = C[x].values - np.dot(cvar, beta_x)
        if y_covar is not None:
            cvar = np.atleast_2d(C[y_covar].values)
            beta_y = np.linalg.lstsq(cvar, C[y].values, rcond=None)[0]
            res_y = C[y].values - np.dot(cvar, beta_y)

    return _bicorr(res_x, res_y, method=method, tail=tail)


def pcm(df):
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
    V = df.cov()  # Covariance matrix
    Vi = np.linalg.pinv(V)  # Inverse covariance matrix
    D = np.diag(np.sqrt(1 / np.diag(Vi)))
    pcor = -1 * (D @ Vi @ D)  # Partial correlation matrix
    pcor[np.diag_indices_from(pcor)] = 1
    return pd.DataFrame(pcor, index=V.index, columns=V.columns)
