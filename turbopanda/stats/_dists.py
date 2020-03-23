#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defining different distributions."""

import string
from scipy import stats
import pandas as pd


def _slim_continuous_dists():
    return (
        'chi2', 'norm', 'expon', 't', 'logistic', 'gumbel_l', 'gumbel_r', 'loggamma', 'gamma'
    )


def _scipy_continuous_distributions():
    """Returns the scipy continuous distribution set."""
    return [s for s in dir(stats) if
            not s.startswith("_") and
            ("rv_" not in s) and
            (s[0] in string.ascii_lowercase) and
            hasattr(getattr(stats, s), 'fit')]


def _scipy_discrete_distributions():
    """Returns the scipy discrete distribution set."""
    return [s for s in dir(stats) if
            not s.startswith("_") and
            ("rv_" not in s) and
            (s[0] in string.ascii_lowercase) and
            hasattr(getattr(stats, s), 'pmf')]


def _fit_cont_dist(name, data):
    """Given name and data, fit parameters to the distribution."""
    dn = getattr(stats, name)
    params = dn.fit(data)
    # return fitted model to this.
    return dn(*params)


def _get_shape_arg(name):
    """Given name of distribution, get number of parameters"""
    return getattr(stats, name).numargs


def auto_fit(data, option="slim"):
    """Attempts to fit the best distribution to some given data.

    Currently only compatible with continuous distributions with a *fit* method.

    Parameters
    ----------
    data : np.ndarray (n,)
        If multi-dimensional, flattens it to 1D.
    option : str/tuple of str, default="slim"
        Choose between "slim" and "full", slim only compares the most popular distributions
        with fewer parameters.

    Returns
    -------
    df : DataFrame
        Results dataframe of each fitted model.
    """
    if data.ndim > 1:
        data = data.flatten()

    if isinstance(option, str):
        opt_m = {'slim': _slim_continuous_dists, 'full': _scipy_continuous_distributions}
        dists = opt_m[option]()
    elif isinstance(option, (tuple, list)):
        dists = option
    else:
        raise TypeError("`option` must be of type [str, list, tuple], not {}".format(type(option)))

    # use qqplot to find best conditions
    _param_scores = [stats.probplot(data, dist=_fit_cont_dist(d, data))[-1] for d in dists]
    # make as dataframe
    df = pd.DataFrame(_param_scores, columns=['slope','intercept','r'], index=dists)
    df['shape_args'] = [_get_shape_arg(d) for d in dists]
    # correction for R maybe useful
    df['r_corrected'] = (df['r']**2) - (df['shape_args']/16.)
    return df
