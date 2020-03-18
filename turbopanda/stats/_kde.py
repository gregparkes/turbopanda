#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generates 1D kernel density approximations for continuous AND discrete applications."""

import numpy as np
from typing import Optional
from scipy.interpolate import make_interp_spline
from scipy import stats
from scipy import optimize as so


def _iqr(a):
    """Calculate the IQR for an array of numbers."""
    return stats.scoreatpercentile(np.asarray(a), 75) - stats.scoreatpercentile(np.asarray(a), 25)


def _smooth_kde(_x, _bins, _mt, params):
    xn = np.linspace(_bins.min(), _bins.max(), 300)
    spl = make_interp_spline(_bins, _mt.pmf(_bins, *params), 2)
    return xn, spl(xn)


""" Discrete distribution likelihood functions """


def _clean_binomial(theta):
    """Rounds down appropriate variables for n, p, loc."""
    return np.round(theta[0]), theta[1], np.round(theta[2])


def _negative_binomial_loglikelihood(theta, x):
    """where p is a list of parameters, x is the input space"""
    n, p, loc = _clean_binomial(theta)
    return -1 * (stats.binom.logpmf(x, n, p, loc)).sum()


def _negative_negbinomial_loglikelihood(theta, x):
    n, p, loc = _clean_binomial(theta)
    return -1 * (stats.nbinom.logpmf(x, n, p, loc)).sum()


""" Miscallaenous function for bin optimization """


def freedman_diaconis_bins(a: np.ndarray) -> int:
    """
    Calculate number of hist bins using Freedman-Diaconis rule.

    Taken from https://github.com/mwaskom/seaborn/blob/master/seaborn/distributions.py
    """
    # From https://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    if len(a) < 2:
        return 1
    h = 2 * _iqr(a) / (a.shape[0] ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((np.nanmax(a) - np.nanmin(a)) / h))


def get_bins(x):
    """Gets the optimal number of bins for a continuous or discrete set of data."""
    # if X is float, use freedman_diaconis_bins determinant, else simply np.arange for integer input.
    if x.dtype.kind == 'f' or (np.nanmax(x) - np.nanmin(x) > 100):
        bin_n = min(freedman_diaconis_bins(x), 50)
        _, bins = np.histogram(x, bin_n)
    elif (x.dtype.kind == 'i' or x.dtype.kind == 'u') and (np.nanmax(x) - np.nanmin(x) <= 100):
        # firstly we determine if the range is small, because if not we just use the above technique
        bins = np.arange(np.nanmin(x), np.nanmax(x) + 2, step=1)
    else:
        raise TypeError("np.ndarray type '{}' not recognized; must be float or int".format(x.dtype))
    return bins


""" Calculating the KDE distributions for continuous and discrete distributions"""


def univariate_kde(X: np.ndarray,
                   bins: Optional[np.ndarray] = None,
                   kde_name: Optional[str] = 'norm',
                   kde_range: float = 1e-3,
                   smoothen_kde: bool = True,
                   verbose: int = 0,
                   return_dist : bool = False):
    """Determines a univariate KDE approximation on a 1D sample, either continuous or discrete.

    .. note:: If x is multi-dimensional, the array is flattened.

    Parameters
    ----------
    X : np.ndarray
        The 1D set to calculate. If more than 1-dim it is flattened.
    bins : np.ndarray, optional
        A range relating to the desired number of bins
    kde_name : str, optional, default='norm'
        The name relating to a scipy.stats.<kde_name> distribution
    kde_range : float, default=1e-3
        A range to use for continuous distributions
    smoothen_kde : bool, default=True
        whether to smooth the distribution appearance for discrete applications
    verbose : int, optional, default=0
        If > 0, prints out useful messages
    return_dist : bool, default=False
        If True, returns the frozen scipy.stats.<kde_name> model with fitted parameters

    Returns
    -------
    x_kde : np.ndarray
        The x-coordinates of the KDE
    y_kde : np.ndarray
        The kernel density approximation as a density score
    """
    # convert to numpy
    _X = np.asarray(X)
    if _X.ndim > 1:
        _X = _X.flatten()
    if bins is None:
        bins = get_bins(_X)

    # get the scipy.stats module
    _mt = getattr(stats, kde_name)

    # these functions directly approximate the parameter to use for these distributions
    discrete_kde_single_ = {
        'bernoulli': np.mean, 'poisson': np.mean, 'geom': lambda x: 1. / np.mean(x)
    }
    # these functions approximate a close initial value problem (IVP) to solve.
    discrete_kde_ests_ = {
        # function to calculate + estimate initial parameters for n, p, loc
        "binom": (_negative_binomial_loglikelihood, _clean_binomial,
                  np.nanmax, lambda x: (np.bincount(x).argmax() + 1) / np.nanmax(x), np.nanmin),
        "nbinom": (_negative_negbinomial_loglikelihood, _clean_binomial,
                   lambda x: np.nanmean(x)/2., lambda x: (np.bincount(x).argmax() + 1) / np.nanmax(x), np.nanmin)
    }

    if hasattr(_mt, "fit"):
        # fit mt with x
        params = _mt.fit(_X)
        model = _mt(*params)
        # generate x_kde
        x_kde = np.linspace(model.ppf(kde_range), model.ppf(1 - kde_range), 200)
        y_kde = model.pdf(x_kde)
    else:
        params = []
        # try to fit a model??
        if kde_name in discrete_kde_single_.keys():
            # single parameter discrete models - which take the data and get the 'mean' or whatever
            params = [discrete_kde_single_[kde_name](_X)]
        elif kde_name in discrete_kde_ests_.keys():
            # 1 is loglikelihood function, 2 is cleaner function, 3:k are parameters
            var_ = discrete_kde_ests_[kde_name]
            # estimates using given functions
            ests_ = [v(_X) for v in var_[2:]]
            # minimization function
            params = var_[1](so.fmin(var_[0], ests_, args=(_X,), ftol=1e-4, full_output=True, disp=False)[0])
            if verbose:
                print("estimated: {}, determined: {}".format(ests_, params))
        else:
            raise ValueError("discrete kde '{}' not found in {}".format(_kde_name,
                                                                        list(discrete_kde_single_.keys()) + list(
                                                                            discrete_kde_ests_.keys())))
        model = _mt(*params)
        # choice of smoothing..
        if smoothen_kde:
            x_kde, y_kde = _smooth_kde(_X, bins, _mt, params)
        else:
            x_kde, y_kde = bins, _mt.pmf(bins, *params)

    if return_dist:
        return x_kde, y_kde, model
    else:
        return x_kde, y_kde
