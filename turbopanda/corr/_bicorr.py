""" Internal bivariate methods. """

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr

from ._corr_metrics import percbend, shepherd, skipped
from ._stats_extra import compute_esci, power_corr

from turbopanda.stats._lmfast import lm
from turbopanda.utils import belongs, instance_check, \
    is_column_boolean, is_column_float, remove_na, union, is_dataframe_float, \
    bounds_check


__all__ = ('bicorr', 'partial_bicorr')


"""Methods to handle continuous-continuous, continuous-boolean and boolean-boolean cases of correlation. """


def _both_continuous(x, y):
    return is_column_float(x) and is_column_float(y)


def _continuous_bool(x, y):
    return is_column_float(x) and is_column_boolean(y)


def _bool_continuous(x, y):
    return is_column_boolean(x) and is_column_float(y)


def _boolbool(x, y):
    return is_column_boolean(x) and is_column_boolean(y)


""" Main inner method for bi-correlation """


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
    # use linear model to generate predictions
    px, r_x = lm(_data[covar], _data[x])
    py, r_y = lm(_data[covar], _data[y])
    # wrap residuals as series
    # if one is a boolean operation, we must preserve structure
    res_x = pd.Series(r_x, name=x)
    res_y = pd.Series(r_y, name=y)
    """ Perform bivariate correlate as normal. """
    # calculate bicorrelation on residuals
    return _bicorr_inner(res_x, res_y, method=method, tail=tail, verbose=0)


""" Public method """


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
