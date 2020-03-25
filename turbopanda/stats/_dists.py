#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defining different distributions."""

import string
import numpy as np
import pandas as pd
from scipy import stats

from ._kde import fit_model


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


def _get_shape_arg(name):
    """Given name of distribution, get number of parameters"""
    return getattr(stats, name).numargs


def _get_distribution_set(option):
    if isinstance(option, str):
        opt_m = {'slim': _slim_continuous_dists, 'full': _scipy_continuous_distributions,
                 'auto': _slim_continuous_dists}
        return opt_m[option]()
    elif isinstance(option, (list, tuple)):
        return option
    else:
        raise TypeError("`option` must be of type [str, list, tuple], not {}".format(type(option)))


def _get_qqplot_score_correlate(data, distributions):
    score_list = [stats.probplot(data, dist=fit_model(data, d))[-1] for d in distributions]
    df = pd.DataFrame(score_list, columns=('slope', 'intercept', 'r'), index=distributions)
    df['shape_args'] = [_get_shape_arg(d) for d in distributions]
    # correction for R maybe useful
    df['r_corrected'] = (df['r'] ** 2) - (df['shape_args'] / 16.)
    return df


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
    _data = np.asarray(data)

    if _data.ndim > 1:
        _data = _data.flatten()

    dists = _get_distribution_set(option)
    # make as dataframe
    return _get_qqplot_score_correlate(_data, dists)


def auto_fit_voted(dataset, option="slim"):
    """Fits the best distribution to a group of $k$-vectors, assuming they all belong to the same distribution.

    This is mechanism by majority voting between distributions.

    Parameters
    ----------
    dataset : list of np.ndarray (n,)/pd.Series or pd.DataFrame (n,k)
        The data columns to calculate on.
    """
    return NotImplemented
