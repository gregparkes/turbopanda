#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty histograms in primitive matplotlib."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from typing import Union, Tuple, List, Optional

from turbopanda.utils import instance_check
from turbopanda.stats import univariate_kde, get_bins

""" Helper methods for redundant code such as plotting, getting bin type, smoothing etc. """


def _plot_hist(_x, _ax, *args, **kwargs):
    if _x.dtype.kind == 'f':
        _ = _ax.hist(_x, *args, **kwargs)
    elif _x.dtype.kind == 'i' or _x.dtype.kind == 'u':
        # make sure to align all the bars to the left in this case
        _ = _ax.hist(_x, align='left', *args, **kwargs)
    else:
        raise TypeError("np.ndarray type '{}' not recognized; must be float or int".format(_x.dtype))


""" The meat and bones """


def histogram(X: Union[np.ndarray, pd.Series, List, Tuple],
              bins: Optional[Union[int, np.ndarray]] = None,
              density: bool = True,
              kde: str = "norm",
              stat: bool = False,
              ax=None,
              x_label: str = "",
              title: str = "",
              kde_range: float = 1e-3,
              smoothen_kde: bool = True,
              verbose: int = 0,
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
    verbose : int, default=0
        If > 0, prints out useful messages
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
    instance_check(stat, bool)
    instance_check(title, str)
    instance_check(x_label, str)
    instance_check(kde, (str, type(None)))
    instance_check(kde_range, float)
    instance_check(smoothen_kde, bool)

    # convert to numpy.
    _X = np.asarray(X)
    if _X.ndim > 1:
        _X = _X.flatten()
    # make bins if set to None
    if bins is None:
        # if X is float, use freedman_diaconis_bins determinant, else simply np.arange for integer input.
        bins = get_bins(_X)

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
            x_kde, y_kde = univariate_kde(_X, bins, kde,
                                          kde_range=1e-3, smoothen_kde=smoothen_kde,
                                          verbose=verbose)
            # plot
            ax.plot(x_kde, y_kde, "-", color='r')
        else:
            raise ValueError("kde value '{}' not found in scipy.stats".format(kde))

    return ax
