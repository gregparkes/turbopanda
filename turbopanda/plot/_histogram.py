#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty histograms in primitive matplotlib."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from turbopanda.utils import instance_check


def _iqr(a):
    """Calculate the IQR for an array of numbers."""
    return stats.scoreatpercentile(np.asarray(a), 75) - stats.scoreatpercentile(np.asarray(a), 25)


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


def histogram(X,
              bins=None,
              density=True,
              kde=True,
              stat=False,
              full_decor=False,
              ax=None,
              x_label=None,
              title="",
              *hist_args,
              **hist_kwargs):
    """Draws pretty histograms using `X`.

    Parameters
    ----------
    X : np.ndarray/pd.Series (1d)
        The data column to draw.
    bins : int, optional
        If None, uses optimal algorithm to find best bin count
    density : bool
        If True, uses density approximation
    kde : bool
        If True, uses a kernel density approximation, and uses `normed`
    stat : bool
        If True, sets statistical variables in legend
    full_decor : bool
        If True, draws the complete KDE distribution with mean/sd lines to +- 3SD
    ax : matplotlib.ax object, optional
        If None, creates one.
    x_label : str, optional
        If None, uses `x-axis`.
    title : str, optional
        If None, uses `Default Title`

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
    instance_check(kde, bool)
    instance_check(title, str)

    # convert to numpy.
    _X = np.asarray(X)
    # do checks
    if bins is None:
        bins = min(freedman_diaconis_bins(_X), 50)
    if kde:
        density = True
    # plot histogram
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if x_label is None:
        if isinstance(X, pd.Series) and X.name is not None:
            x_label = X.name
        else:
            x_label = "x-axis"

    # fit to a normal distribution...
    mean, sd = stats.norm.fit(_X)

    if stat:
        stat_label = "mean: {:0.2f}, sd: {:0.3f},\n skew: {:0.3f} kurt: {:0.3f}".format(
            mean, sd, stats.skew(_X), stats.kurtosis(_X))
        _ = ax.hist(_X, bins=bins, density=density, label=stat_label, *hist_args, **hist_kwargs)
        ax.legend()
    else:
        _ = ax.hist(_X, bins=bins, density=density, rwidth=.9, *hist_args, **hist_kwargs)

    ax.set_xlabel(x_label)
    ax.set_title(title)

    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Counts")

    if kde:
        x_kde = np.linspace(_X.min() - sd, _X.max() + sd, 200)
        # generate y kde
        y_kde = stats.norm.pdf(x_kde, loc=mean, scale=sd)
        # plot
        ax.plot(x_kde, y_kde, "-", color='r')
        if full_decor:
            ax.vlines(mean, 0, y_kde.max(), linestyle="-", color='r')
            ax.vlines([mean-sd, mean+sd], 0, y_kde[np.argmin(np.abs(x_kde-mean-sd))], linestyle="-", color='#E89E30')
            ax.vlines([mean - 2*sd, mean + 2*sd], 0, y_kde[np.argmin(np.abs(x_kde - mean - 2*sd))], linestyle="-",
                      color='#FFF700')
            ax.vlines([mean - 3 * sd, mean + 3 * sd], 0, y_kde[np.argmin(np.abs(x_kde - mean - 3 * sd))], linestyle="-",
                      color='#68B60E')

    return ax
