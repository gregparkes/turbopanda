""" Internal bivariate methods. """

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr

from typing import List, Tuple, Union

from ._corr_metrics import percbend, shepherd, skipped, corr_ratio, kramers_v
from ._stats_extra import compute_esci, power_corr

from turbopanda.stats._lmfast import lm
from turbopanda.utils import (
    belongs,
    instance_check,
    is_column_boolean,
    is_column_float,
    is_column_int,
    is_column_object,
    remove_na,
    union,
    is_dataframe_float,
)

__all__ = ("bicorr", "partial_bicorr")

"""Methods to handle continuous-continuous,
    continuous-boolean and
    boolean-boolean cases of correlation. """


def _both_continuous(x, y):
    return (x.dtype.kind == 'f') and (y.dtype.kind == 'f')


def _both_integers(x, y):
    return is_column_int(x) and is_column_int(y)


def _continuous_bool(x, y):
    return (x.dtype.kind == 'f') and is_column_boolean(y)


def _bool_continuous(x, y):
    return is_column_boolean(x) and (y.dtype.kind == 'f')


def _continuous_categorical(x, y):
    return (x.dtype.kind == 'f') and is_column_object(y)


def _categorical_continuous(x, y):
    return is_column_object(x) and (y.dtype.kind == 'f')


def _both_categorical(x, y):
    return is_column_object(x) and is_column_object(y)


def _both_bool(x, y):
    return is_column_boolean(x) and is_column_boolean(y)


def _preprocess_numpy_pair(x, y):
    xo = np.asarray(x)
    yo = np.asarray(y)
    xo, yo = remove_na(xo, yo, paired=True)
    return xo, yo


def _compute_correlative_all(x, y, xa, ya, method):
    outliers = []
    # where x, y are pd.Series, xa, ya are preprocessed numpy arrays
    if _both_continuous(x, y):
        if method == "pearson":
            r, pval = pearsonr(xa, ya)
        elif method == "spearman":
            r, pval = spearmanr(xa, ya)
        elif method == "kendall":
            r, pval = kendalltau(xa, ya)
        elif method == "percbend":
            r, pval = percbend(xa, ya)
        elif method == "shepherd":
            r, pval, outliers = shepherd(xa, ya)
        elif method == "skipped":
            r, pval, outliers = skipped(xa, ya, method="spearman")
        else:
            raise ValueError("Method not recognized.")

    elif _both_integers(x, y):
        # handle the integer-integer use case.
        r, pval = spearmanr(xa, ya)
    # if they're both categories (strings), then use kramers_v
    elif _continuous_categorical(x, y):
        # correlation ratio [0, 1]
        r, pval = corr_ratio(xa, ya)
    elif _categorical_continuous(x, y):
        # correlation ratio [0, 1]
        r, pval = corr_ratio(ya, xa)
    elif _both_categorical(x, y):
        # kramer's v for categorical-categorical [0, 1]
        r, pval = kramers_v(x, y, True)
    elif _continuous_bool(x, y):
        # sort them into order, it matters
        r, pval = pointbiserialr(xa, ya.astype(np.uint8))
    elif _bool_continuous(x, y):
        # sort them into order, it matters
        r, pval = pointbiserialr(xa.astype(np.uint8), ya)
    elif _both_bool(x, y):
        # use spearman
        r, pval = spearmanr(xa.astype(np.uint8), ya.astype(np.uint8))
    else:
        raise TypeError(
            "columns '{}':{} to '{}':{} combination not accepted for `bicorr`.".format(
                x.name, x.dtype, y.name, y.dtype
            )
        )
    assert not np.isnan(r), "Correlation returned NaN. Check your data."
    return r, pval, outliers


""" Main inner method for bi-correlation """


def _bicorr_inner_score(x: pd.Series,
                        y: pd.Series,
                        method: str = "spearman"):
    """Internal method for bicorrelation here"""
    # convert to numpy
    x_arr, y_arr = _preprocess_numpy_pair(x, y)
    r, _, _ = _compute_correlative_all(x, y, x_arr, y_arr, method)
    return {
        "x": x.name, "y": y.name, "r": r
    }


def _bicorr_inner_full(x: pd.Series,
                       y: pd.Series,
                       method: str = "spearman",
                       tail: str = "two-sided",
                       rounding_factor: int = 3):
    """Internal method for bicorrelation here"""
    # convert to numpy
    x_arr, y_arr = _preprocess_numpy_pair(x, y)
    nx = x_arr.size
    # computes basic r and pvalue
    r, pval, outliers = _compute_correlative_all(x, y, x_arr, y_arr, method)

    # Compute r2 and adj_r2
    r2 = r ** 2
    adj_r2 = 1 - (((1 - r2) * (nx - 1)) / (nx - 3))

    # Compute the parametric 95% confidence interval and power
    if r2 < 1:
        ci = compute_esci(stat=r, nx=nx, ny=nx, eftype="r")
        pr = round(power_corr(r=r, n=nx, power=None, alpha=0.05, tail=tail), 3)
    else:
        ci = [1.0, 1.0]
        pr = np.inf

    # Create dictionary
    return {
        "x": x.name,
        "y": y.name,
        "n": nx,
        "method": method,
        "r": round(r, rounding_factor),
        "r2": round(r2, rounding_factor),
        "adj_r2": round(adj_r2, rounding_factor),
        "CI95_lower": ci[0],
        "CI95_upper": ci[1],
        "p_val": pval if tail == "two-sided" else 0.5 * pval,
        "power": pr,
        "outliers": sum(outliers) if method in ("shepherd", "skipped") else np.nan,
    }


def _partial_bicorr_inner(
        data: pd.DataFrame,
        x,
        y,
        covar,
        tail: str = "two-sided",
        method: str = "spearman",
        output: str = "score"
):
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

    if output == "score":
        return _bicorr_inner_score(res_x, res_y, method)
    else:
        return _bicorr_inner_full(res_x, res_y, method=method, tail=tail)


""" #########  Public method ############### """


def bicorr(
        x: pd.Series,
        y: pd.Series,
        method: str = "spearman",
        tail: str = "two-sided",
        output: str = "score"
) -> Union[float, dict]:
    """(Robust) correlation between two variables.

    Adapted from the `pingouin` library, made by Raphael Vallat.

    .. [1] https://github.com/raphaelvallat/pingouin/blob/master/pingouin/correlation.py

    Parameters
    ----------
    x, y : pd.Series
        First and second set of observations. x and y must be independent.
    method : str
        Specify which method to use for the computation of the correlation
        coefficient. Available methods are ::
        'pearson' : Pearson product-moment correlation
        'spearman' : Spearman rank-order correlation
        'kendall' : Kendall’s tau (ordinal data)
        'biserial' : Biserial correlation (continuous and boolean data)
        'percbend' : percentage bend correlation (robust)
        'shepherd' : Shepherd's pi correlation (robust Spearman)
        'skipped' : skipped correlation (robust Spearman, requires sklearn)
    tail : str
        Specify whether to return 'one-sided' or 'two-sided' p-value.
    output : str, default='score'
        Determines whether to display the full output or
            just the correlation (r) score
            options are {'score', 'full'}.

    Returns
    -------
    stats : float/dict
        Test summary ::
        'n' : Sample size (after NaN removal)
        'outliers' : number of outliers (only for 'shepherd' or 'skipped')
        'r' : Correlation coefficient
        'CI95' : 95% parametric confidence intervals
        'r2' : R-squared
        'adj_r2' : Adjusted R-squared
        'method' : pearson/spearman/biserial... etc
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
    belongs(output, ('score', 'full'))
    # Check size
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")

    if output == "score":
        return _bicorr_inner_score(x, y, method)
    else:
        return _bicorr_inner_full(x, y, method, tail=tail)


def partial_bicorr(
        data: pd.DataFrame,
        x: str,
        y: str,
        covar: Union[str, List[str], Tuple[str, ...], pd.Index],
        method: str = "spearman",
        tail: str = "two-sided",
        output: str = 'score'
) -> Union[float, dict]:
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
            covar must be made of continuous columns.
            If x, y are not continuous, will perform logistic regression
            to generate residuals.
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
    tail : string
        Specify whether to return the 'one-sided' or 'two-sided' p-value.
    output : str, default='score'
        Determines whether to display the full output or
            just the correlation (r) score
            options are {'score', 'full'}.

    Returns
    -------
    stats : float/dict
        Test summary ::
        'n' : Sample size (after NaN removal)
        'outliers' : number of outliers (only for 'shepherd' or 'skipped')
        'r' : Correlation coefficient
        'CI95' : 95% parametric confidence intervals
        'r2' : R-squared
        'adj_r2' : Adjusted R-squared
        'method' : pearson/spearman/biserial... etc
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
    belongs(tail, ("one-sided", "two-sided"))
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
    belongs(output, ('score', 'full'))
    # perform a check to make sure every column in `covar`
    # is continuous.
    if not is_dataframe_float(data[covar]):
        raise TypeError(
            "`covar` variables in `partial_bicorr` "
            "all must be of type `float`/continuous."
        )

    return _partial_bicorr_inner(data, x, y, covar, tail=tail,
                                 method=method, output=output)
