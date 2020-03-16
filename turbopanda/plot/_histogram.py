#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty histograms in primitive matplotlib."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.interpolate import make_interp_spline
from typing import Union, Tuple, List, Optional

from turbopanda.utils import instance_check


def _iqr(a):
    """Calculate the IQR for an array of numbers."""
    return stats.scoreatpercentile(np.asarray(a), 75) - stats.scoreatpercentile(np.asarray(a), 25)


def _plot_hist(_x, _ax, *args, **kwargs):
    if _x.dtype.kind == 'f':
        _ = _ax.hist(_x, *args, **kwargs)
    elif _x.dtype.kind == 'i' or _x.dtype.kind == 'u':
        # make sure to align all the bars to the left in this case
        _ = _ax.hist(_x, align='left', *args, **kwargs)
    else:
        raise TypeError("np.ndarray type '{}' not recognized; must be float or int".format(_x.dtype))


def _get_bins(_x):
    # if X is float, use freedman_diaconis_bins determinant, else simply np.arange for integer input.
    if _x.dtype.kind == 'f' or (np.max(_x) - np.min(_x) > 50):
        bins = min(freedman_diaconis_bins(_x), 50)
    elif (_x.dtype.kind == 'i' or _x.dtype.kind == 'u') and np.max(_x) - np.min(_x) <= 50:
        # firstly we determine if the range is small, because if not we just use the above technique
        bins = np.arange(np.min(_x), np.max(_x) + 2, step=1)
    else:
        raise TypeError("np.ndarray type '{}' not recognized; must be float or int".format(_x.dtype))
    return bins


#def _negative_discrete_likelihood()


def _get_kde(_x, _kde_name, _bins,
             discrete_options_, discrete_est_options_,
             _kde_range=1e-3,
             _smoothen_kde=True):
    # if _mt doesn't have the `fit` attribute, its discrete and we use bins as x_kde
    _mt = getattr(stats, _kde_name)

    if hasattr(_mt, "fit"):
        # fit mt with x
        params = _mt.fit(_x)
        model = _mt(*params)
        # generate x_kde
        x_kde = np.linspace(model.ppf(_kde_range), model.ppf(1 - _kde_range), 200)
        return x_kde, model.pdf(x_kde)
    else:
        # try to fit a model??
        if _kde_name in discrete_options_.keys():
            # single parameter discrete models - which take the data and get the 'mean' or whatever
            param = [discrete_options_[_kde_name](_x)]
            # x_kde is the bins.
            if _smoothen_kde:
                xn = np.linspace(_x.min(), _x.max()+1, 300)
                spl = make_interp_spline(_bins, _mt.pmf(_bins, *param), 2)
                return xn, spl(xn)
            else:
                return _bins, _mt.pmf(_bins, *param)
        elif _kde_name in discrete_est_options_.keys():
            # binomial distribution case where we estimate parameters
            param_names = _mt.shapes.split(", ")


        else:
            raise ValueError("discrete kde '{}' not found in {}".format(_kde_name, tuple(discrete_options_.keys())))


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


def histogram(X: Union[np.ndarray, pd.Series, List, Tuple],
              bins: Optional[Union[int, np.ndarray]] = None,
              density: bool = True,
              kde: str = "norm",
              stat: bool = False,
              full_decor: bool = False,
              ax=None,
              x_label: str = "",
              title: str = "",
              kde_range: float = 1e-3,
              smoothen_kde: bool = True,
              *hist_args,
              **hist_kwargs):
    """Draws pretty histograms using `X`.

    Parameters
    ----------
    X : np.ndarray/pd.Series (1d)
        The data column to draw.
    bins : int, optional
        If None, uses optimal algorithm to find best bin count
    density : bool, default=True
        If True, uses density approximation
    kde : str, optional, default="norm"
        If None, does not draw a KDE plot
        else, choose from available distributions in `scipy.stats`
    stat : bool, default=False
        If True, sets statistical variables in legend
    full_decor : bool, default=False
        If True, draws the complete KDE distribution with mean/sd lines to +- 3SD
    ax : matplotlib.ax object, optional, default=None
        If None, creates one.
    x_label : str, optional, default=None
        If None, uses `x-axis`.
    title : str, optional, default=""
        If None, uses `Default Title`
    kde_range : float, default=1e-3
        Defines the precision on the KDE range if plotted between (1e-3, 1-1e-3)
        Must be > 0.
    smoothen_kde : bool, default=True
        If discrete-distribution, applies smoothing function to KDE if True

    Other Parameters
    ----------------
    args ; list
        Arguments to pass to `ax.hist`
    kwargs : dict
        Keyword arguments to pass to `ax.hist`

    Returns
    -------
    ax : matplotlib.ax object
        Allows further modifications to the axes post-histogram
    """
    instance_check(X, (np.ndarray, pd.Series, list, tuple))
    instance_check(density, bool)
    instance_check(full_decor, bool)
    instance_check(stat, bool)
    instance_check(title, str)
    instance_check(x_label, str)
    instance_check(kde, (str, type(None)))
    instance_check(kde_range, float)

    # mapping certain discrete kdes to calculate their parameter(s)
    discrete_kde_ = {
        'bernoulli': np.mean, 'poisson': np.mean, 'geom': lambda x: 1./np.mean(x)
    }
    discrete_kde_ests_ = {
        # estimate initial parameters for n, p
        "binom": [np.max, lambda x: (np.bincount(x).argmax()+1) / np.max(x)]
    }

    # convert to numpy.
    _X = np.asarray(X)
    # make bins if set to None
    if bins is None:
        # if X is float, use freedman_diaconis_bins determinant, else simply np.arange for integer input.
        bins = _get_bins(_X)

    if kde:
        density = True
    # plot histogram
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if x_label == "":
        if isinstance(X, pd.Series) and X.name is not None:
            x_label = X.name
        else:
            x_label = "x-axis"

    if stat:
        stat_label = "mean: {:0.2f}, sd: {:0.3f},\n skew: {:0.3f} kurt: {:0.3f}".format(
            np.nanmean(_X), np.nanstd(_X), stats.skew(_X), stats.kurtosis(_X))
        # plot the histogram
        _plot_hist(_X, ax, bins=bins, density=density, rwidth=.9, label=stat_label, *hist_args, **hist_kwargs)
        ax.legend()
    else:
        # plot the histogram
        _plot_hist(_X, ax, bins=bins, density=density, rwidth=.9, *hist_args, **hist_kwargs)

    ax.set_xlabel(x_label)
    ax.set_title(title)

    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Counts")

    if kde is not None:
        if hasattr(stats, kde):
            # fetches the kde if possible
            x_kde, y_kde = _get_kde(_X, _kde_name=kde, _bins=bins,
                                    discrete_options_=discrete_kde_, _kde_range=1e-3, _smoothen_kde=smoothen_kde)
            # plot
            ax.plot(x_kde, y_kde, "-", color='r')
            if full_decor:
                ax.vlines(mean, 0, y_kde.max(), linestyle="-", color='r')
                ax.vlines([mean - sd, mean + sd], 0, y_kde[np.argmin(np.abs(x_kde - mean - sd))], linestyle="-",
                          color='#E89E30')
                ax.vlines([mean - 2 * sd, mean + 2 * sd], 0, y_kde[np.argmin(np.abs(x_kde - mean - 2 * sd))],
                          linestyle="-",
                          color='#FFF700')
                ax.vlines([mean - 3 * sd, mean + 3 * sd], 0, y_kde[np.argmin(np.abs(x_kde - mean - 3 * sd))],
                          linestyle="-",
                          color='#68B60E')
        else:
            raise ValueError("kde value '{}' not found in scipy.stats".format(kde))

    return ax
