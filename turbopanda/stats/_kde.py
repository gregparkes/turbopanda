#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generates 1D kernel density approximations for continuous AND discrete applications."""

from typing import Optional

import numpy as np
from scipy import optimize as so, stats
from scipy.interpolate import make_interp_spline


def _smooth_kde(_bins, _mt, _n=300):
    xn = np.linspace(_bins.min(), _bins.max(), _n)
    spl = make_interp_spline(_bins, _mt.pmf(_bins), 2)
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
    h = 2 * (stats.scoreatpercentile(a, 75) - stats.scoreatpercentile(a, 25)) / (a.shape[0] ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((np.max(a) - np.min(a)) / h))


def get_bins(x):
    """Gets the optimal number of bins for a continuous or discrete set of data."""
    # if X is float, use freedman_diaconis_bins determinant, else simply np.arange for integer input.
    xmax = np.max(x)
    xmin = np.min(x)

    if x.dtype.kind == 'f' or (xmax - xmin > 100):
        bin_n = min(freedman_diaconis_bins(x), 50)
        _, bins = np.histogram(x, bin_n)
    elif (x.dtype.kind == 'i' or x.dtype.kind == 'u') and (xmax - xmin <= 100):
        # firstly we determine if the range is small, because if not we just use the above technique
        bins = np.arange(xmin, xmax + 2, step=1)
    else:
        raise TypeError("np.ndarray type '{}' not recognized; must be float or int".format(x.dtype))
    return bins


""" Calculating the KDE distributions for continuous and discrete distributions"""


def _get_discrete_single():
    """These models have a single parameter and are easily fitted."""
    return {
        'bernoulli': np.mean, 'poisson': np.mean, 'geom': lambda x: 1. / np.mean(x)
    }


def _get_discrete_multiple():
    """These models have more than one parameter, and require log-likelihood maximization; which is expensive."""
    return {
        # function to calculate + estimate initial parameters for n, p, loc
        "binom": (_negative_binomial_loglikelihood, _clean_binomial,
                  np.nanmax, lambda x: (np.bincount(x).argmax() + 1) / np.nanmax(x), np.nanmin),
        "nbinom": (_negative_negbinomial_loglikelihood, _clean_binomial,
                   lambda x: np.nanmean(x) / 2., lambda x: (np.bincount(x).argmax() + 1) / np.nanmax(x), np.nanmin)
    }


def _problem_distributions():
    return ('argus', 'betaprime', 'erlang', 'cosine', 'exponnorm', 'foldcauchy',
            'foldnorm', 'genexpon', 'gausshyper', 'invgauss', 'levy_stable',
            'ksone', 'ncf', 'nct', 'ncx2', 'norminvgauss', 'powerlognorm',
            'rdist', 'recipinvgauss', 'rv_continuous', 'rv_histogram',
            'skewnorm', 'tukeylambda', 'wrapcauchy', 'semicircular', 'vonmises',
            'vonmises_line')


def fit_model(X, name, verbose=0, return_params=False):
    """Given distribution name and X, return a fitted model with it's parameters."""
    _model = getattr(stats, name)
    # this should work.
    if hasattr(_model, "fit"):
        # it's continuous - use the fit method
        params = _model.fit(X)
        _fitted = _model(*params)
    else:
        # it's discrete, damn!
        if name in _get_discrete_single():
            # nice and easy.
            params = [_get_discrete_single()[name](X)]
        elif name in _get_discrete_multiple():
            # this is where the fun begins...
            # 0 is loglikelihood function, #1 is cleaner function, 2:k are parameters
            var_ = _get_discrete_multiple()[name]
            # guess good starting locations for optimization.
            inits_ = [v(X) for v in var_[2:]]
            # minimization function
            params = var_[1](so.fmin(var_[0], inits_, args=(X,), ftol=1e-4, full_output=True, disp=False)[0])
            if verbose > 0:
                print("initial guess: {}, minimized guess: {}".format(inits_, params))
        else:
            raise ValueError("discrete kde '{}' not found in {}".format(name, list(_get_discrete_single()) + list(
                _get_discrete_multiple())))
        _fitted = _model(*params)

    if return_params:
        return _fitted, params
    else:
        return _fitted


def univariate_kde(X: np.ndarray,
                   bins: Optional[np.ndarray] = None,
                   kde_name: str = 'norm',
                   kde_range: float = 1e-3,
                   smoothen_kde: bool = True,
                   verbose: int = 0,
                   return_dist: bool = False):
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
        If 'freeform': fits the best KDE to the data points.
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

    supported_disc_dists = list(_get_discrete_single()) + list(_get_discrete_multiple())
    # convert to numpy
    _X = np.asarray(X)
    if _X.ndim > 1:
        _X = _X.flatten()
    if bins is None:
        bins = get_bins(_X)

    if kde_name == 'freeform':
        _model = stats.gaussian_kde(_X)
        x_kde = np.linspace(_X.min(), _X.max(), 200)
        y_kde = _model.pdf(x_kde)
    else:
        _model, _params = fit_model(_X, kde_name, verbose=verbose, return_params=True)

        if kde_name in supported_disc_dists:
            if smoothen_kde:
                x_kde, y_kde = _smooth_kde(bins, _model)
            else:
                x_kde, y_kde = bins, _model.pmf(bins)
        else:
            # generate x_kde
            x_kde = np.linspace(_model.ppf(kde_range), _model.ppf(1 - kde_range), 200)
            y_kde = _model.pdf(x_kde)

    if return_dist:
        return x_kde, y_kde, _model
    else:
        return x_kde, y_kde
