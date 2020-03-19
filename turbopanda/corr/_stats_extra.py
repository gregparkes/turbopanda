#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Additional statistical functions. Taken from pingouin library."""
import warnings

import numpy as np
from scipy import stats
from scipy.optimize import brenth
from scipy.stats import norm

__all__ = ("compute_esci", "power_corr")


def compute_esci(stat=None, nx=None, ny=None, paired=False, eftype='cohen',
                 confidence=.95, decimals=2):
    """Parametric confidence intervals around a Cohen d or a
    correlation coefficient.
    Parameters
    ----------
    stat : float
        Original effect size. Must be either a correlation coefficient or a
        Cohen-type effect size (Cohen d or Hedges g).
    nx, ny : int
        Length of vector x and y.
    paired : bool
        Indicates if the effect size was estimated from a paired sample.
        This is only relevant for cohen or hedges effect size.
    eftype : string
        Effect size type. Must be 'r' (correlation) or 'cohen'
        (Cohen d or Hedges g).
    confidence : float
        Confidence level (0.95 = 95%)
    decimals : int
        Number of rounded decimals.
    Returns
    -------
    ci : array
        Desired converted effect size
    Notes
    -----
    To compute the parametric confidence interval around a
    **Pearson r correlation** coefficient, one must first apply a
    Fisher's r-to-z transformation:
    .. math:: z = 0.5 \\cdot \\ln \\frac{1 + r}{1 - r} = \\text{arctanh}(r)
    and compute the standard deviation:
    .. math:: se = \\frac{1}{\\sqrt{n - 3}}
    where :math:`n` is the sample size.
    The lower and upper confidence intervals - *in z-space* - are then
    given by:
    .. math:: ci_z = z \\pm crit \\cdot se
    where :math:`crit` is the critical value of the nomal distribution
    corresponding to the desired confidence level (e.g. 1.96 in case of a 95%
    confidence interval).
    These confidence intervals can then be easily converted back to *r-space*:
    .. math::
        ci_r = \\frac{\\exp(2 \\cdot ci_z) - 1}{\\exp(2 \\cdot ci_z) + 1} =
        \\text{tanh}(ci_z)
    A formula for calculating the confidence interval for a
    **Cohen d effect size** is given by Hedges and Olkin (1985, p86).
    If the effect size estimate from the sample is :math:`d`, then it is
    normally distributed, with standard deviation:
    .. math::
        se = \\sqrt{\\frac{n_x + n_y}{n_x \\cdot n_y} +
        \\frac{d^2}{2 (n_x + n_y)}}
    where :math:`n_x` and :math:`n_y` are the sample sizes of the two groups.
    In one-sample test or paired test, this becomes:
    .. math::
        se = \\sqrt{\\frac{1}{n_x} + \\frac{d^2}{2 \\cdot n_x}}
    The lower and upper confidence intervals are then given by:
    .. math:: ci_d = d \\pm crit \\cdot se
    where :math:`crit` is the critical value of the nomal distribution
    corresponding to the desired confidence level (e.g. 1.96 in case of a 95%
    confidence interval).
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Fisher_transformation
    .. [2] Hedges, L., and Ingram Olkin. "Statistical models for
           meta-analysis." (1985).
    .. [3] http://www.leeds.ac.uk/educol/documents/00002182.htm
    Examples
    --------
    1. Confidence interval of a Pearson correlation coefficient
    >>> import pingouin as pg
    >>> x = [3, 4, 6, 7, 5, 6, 7, 3, 5, 4, 2]
    >>> y = [4, 6, 6, 7, 6, 5, 5, 2, 3, 4, 1]
    >>> nx, ny = len(x), len(y)
    >>> stat = np.corrcoef(x, y)[0][1]
    >>> ci = pg.compute_esci(stat=stat, nx=nx, ny=ny, eftype='r')
    >>> print(stat, ci)
    0.7468280049029223 [0.27 0.93]
    2. Confidence interval of a Cohen d
    >>> import pingouin as pg
    >>> x = [3, 4, 6, 7, 5, 6, 7, 3, 5, 4, 2]
    >>> y = [4, 6, 6, 7, 6, 5, 5, 2, 3, 4, 1]
    >>> nx, ny = len(x), len(y)
    >>> stat = pg.compute_effsize(x, y, eftype='cohen')
    >>> ci = pg.compute_esci(stat=stat, nx=nx, ny=ny, eftype='cohen')
    >>> print(stat, ci)
    0.1537753990658328 [-0.68  0.99]
    """
    # Safety check
    assert eftype.lower() in ['r', 'pearson', 'spearman', 'cohen',
                              'd', 'g', 'hedges']
    assert stat is not None and nx is not None
    assert isinstance(confidence, float)
    assert 0 < confidence < 1

    # Note that we are using a normal dist and not a T dist:
    # from scipy.stats import t
    # crit = np.abs(t.ppf((1 - confidence) / 2), dof)
    crit = np.abs(norm.ppf((1 - confidence) / 2))

    if eftype.lower() in ['r', 'pearson', 'spearman']:
        # Standardize correlation coefficient
        z = np.arctanh(stat)
        se = 1 / np.sqrt(nx - 3)
        ci_z = np.array([z - crit * se, z + crit * se])
        # Transform back to r
        ci = np.tanh(ci_z)
    else:
        if ny == 1 or paired:
            # One sample or paired
            se = np.sqrt(1 / nx + stat ** 2 / (2 * nx))
        else:
            # Two-sample test
            se = np.sqrt(((nx + ny) / (nx * ny)) + (stat ** 2) / (2 * (nx + ny)))
        ci = np.array([stat - crit * se, stat + crit * se])
    return np.round(ci, decimals)


def power_corr(r=None, n=None, power=None, alpha=0.05, tail='two-sided'):
    """
    Evaluate power, sample size, correlation coefficient or
    significance level of a correlation test.
    Parameters
    ----------
    r : float, optional
        Correlation coefficient.
    n : int, optional
        Number of observations (sample size).
    power : float, optional
        Test power (= 1 - type II error).
    alpha : float, optional
        Significance level (type I error probability).
        The default is 0.05.
    tail : str
        Indicates whether the test is "two-sided" or "one-sided".
    Notes
    -----
    Exactly ONE of the parameters ``r``, ``n``, ``power`` and ``alpha`` must
    be passed as None, and that parameter is determined from the others.
    Notice that ``alpha`` has a default value of 0.05 so None must be
    explicitly passed if you want to compute it.
    :py:func:`scipy.optimize.brenth` is used to solve power equations for other
    variables (i.e. sample size, effect size, or significance level). If the
    solving fails, a nan value is returned.
    This function is a mere Python translation of the original `pwr.r.test`
    function implemented in the `pwr` R package.
    All credit goes to the author, Stephane Champely.
    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Hillsdale,NJ: Lawrence Erlbaum.
    .. [2] https://cran.r-project.org/web/packages/pwr/pwr.pdf
    Examples
    --------
    1. Compute achieved power given ``r``, ``n`` and ``alpha``
    >>> from pingouin import power_corr
    >>> print('power: %.4f' % power_corr(r=0.5, n=20))
    power: 0.6379
    2. Compute required sample size given ``r``, ``power`` and ``alpha``
    >>> print('n: %.4f' % power_corr(r=0.5, power=0.80,
    ...                                tail='one-sided'))
    n: 22.6091
    3. Compute achieved ``r`` given ``n``, ``power`` and ``alpha`` level
    >>> print('r: %.4f' % power_corr(n=20, power=0.80, alpha=0.05))
    r: 0.5822
    4. Compute achieved alpha level given ``r``, ``n`` and ``power``
    >>> print('alpha: %.4f' % power_corr(r=0.5, n=20, power=0.80,
    ...                                    alpha=None))
    alpha: 0.1377
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [r, n, power, alpha]])
    if n_none != 1:
        raise ValueError('Exactly one of n, r, power, and alpha must be None')

    # Safety checks
    if r is not None:
        assert -1 <= r <= 1
        r = abs(r)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1
    if n is not None:
        if n <= 4:
            warnings.warn("Sample size is too small to estimate power "
                          "(n <= 4). Returning NaN.")
            return np.nan
    # Define main function
    if tail == 'two-sided':
        def func(_r, _n, _pow, _alpha):
            """Custom function."""
            dof = _n - 2
            ttt = stats.t.ppf(1 - _alpha / 2, dof)
            rc = np.sqrt(ttt ** 2 / (ttt ** 2 + dof))
            zr = np.arctanh(_r) + _r / (2 * (_n - 1))
            zrc = np.arctanh(rc)
            _pow = stats.norm.cdf((zr - zrc) * np.sqrt(_n - 3)) + \
                   stats.norm.cdf((-zr - zrc) * np.sqrt(_n - 3))
            return _pow
    else:
        def func(_r, _n, _pow, _alpha):
            """Custom function option 2."""
            dof = _n - 2
            ttt = stats.t.ppf(1 - _alpha, dof)
            rc = np.sqrt(ttt ** 2 / (ttt ** 2 + dof))
            zr = np.arctanh(_r) + _r / (2 * (_n - 1))
            zrc = np.arctanh(rc)
            _pow = stats.norm.cdf((zr - zrc) * np.sqrt(_n - 3))
            return _pow

    # Evaluate missing variable
    if power is None and n is not None and r is not None:
        # Compute achieved power given r, n and alpha
        return func(r, n, _pow=None, _alpha=alpha)
    elif n is None and power is not None and r is not None:
        # Compute required sample size given r, power and alpha
        def _eval_n(_n, _r, _pow, _alpha):
            return func(_r, _n, _pow, _alpha) - _pow

        try:
            return brenth(_eval_n, 4 + 1e-10, 1e+09, args=(r, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    elif r is None and power is not None and n is not None:
        # Compute achieved r given sample size, power and alpha level
        def _eval_r(_r, _n, _pow, _alpha):
            return func(_r, _n, _pow, _alpha) - _pow

        try:
            return brenth(_eval_r, 1e-10, 1 - 1e-10, args=(n, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha (significance) level given r, n and power
        def _eval_alpha(_alpha, _r, _n, _power):
            return func(_r, _n, _power, _alpha) - _power

        try:
            return brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(r, n, power))
        except ValueError:  # pragma: no cover
            return np.nan
