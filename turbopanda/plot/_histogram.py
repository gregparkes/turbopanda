#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code for plotting pretty histograms in primitive matplotlib."""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from turbopanda.stats import get_bins, univariate_kde, auto_fit
from turbopanda.utils import instance_check

""" Helper methods for redundant code such as plotting, getting bin type, smoothing etc. """


def _plot_hist(_x, _ax, *args, **kwargs):
    if _x.dtype.kind == 'f':
        _ = _ax.hist(_x, *args, **kwargs)
    elif _x.dtype.kind == 'i' or _x.dtype.kind == 'u':
        # make sure to align all the bars to the left in this case
        _ = _ax.hist(_x, align='left', *args, **kwargs)
    else:
        raise TypeError("np.ndarray type '{}' not recognized; must be float or int".format(_x.dtype))


def _criteria_corr_qqplot(r):
    if r > .99:
        return "***"
    elif r > .95:
        return "**"
    elif r > .9:
        return "*"
    elif r > .75:
        return "~"
    else:
        return ".."


def _assign_x_label(title, series_x_l, is_kde, auto_fitted, frozen_dist):
    # this only executes if `x_label` is "".
    if not is_kde:
        # no frozen dist, no autokde
        if series_x_l != '':
            return series_x_l
        elif title is not None:
            return title
        else:
            return "x-axis"
    else:
        if auto_fitted is not None:
            best_model_ = auto_fitted.loc[auto_fitted['r'].idxmax()]
            _crit = _criteria_corr_qqplot(best_model_['r'])
            _args = ", ".join(["{:0.2f}".format(a) for a in frozen_dist.args])
            if series_x_l != '':
                return "{}\n[{}{}({})]".format(series_x_l, frozen_dist.dist.name, _crit, _args)
            else:
                return "{}{}({})".format(frozen_dist.dist.name, _crit, _args)
        else:
            if series_x_l != '':
                return "{}({})".format(series_x_l, frozen_dist.dist.name)
            else:
                return frozen_dist.dist.name


""" The meat and bones """


def histogram(X: Union[np.ndarray, pd.Series, List, Tuple],
              kde: str = "auto",
              bins: Optional[Union[int, np.ndarray]] = None,
              density: bool = True,
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
    kde : str/tuple of str, optional, default="auto"
        If None, does not draw a KDE plot
        If 'auto': attempts to fit the best `continuous` distribution
        If list/tuple: uses 'auto' to fit the best distribution out of options
        else, choose from available distributions in `scipy.stats`
    bins : int, optional
        If None, uses optimal algorithm to find best bin count
    density : bool, default=True
        If True, uses density approximation
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
    instance_check(kde, (str, type(None), list, tuple))
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

    if stat:
        stat_label = "mean: {:0.2f}, sd: {:0.3f},\n skew: {:0.3f} kurt: {:0.3f}".format(
            np.nanmean(_X), np.nanstd(_X), stats.skew(_X), stats.kurtosis(_X))
        # plot the histogram
        _plot_hist(_X, ax, bins=bins, density=density, rwidth=.9, label=stat_label, *hist_args, **hist_kwargs)
        ax.legend()
    else:
        # plot the histogram
        _plot_hist(_X, ax, bins=bins, density=density, rwidth=.9, *hist_args, **hist_kwargs)

    ax.set_title(title)

    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Counts")

    if kde is not None:
        if kde == 'auto' or isinstance(kde, (list, tuple)):
            # uses slim parameters by default
            auto_fitted = auto_fit(_X, kde)
            best_model_ = auto_fitted.loc[auto_fitted['r'].idxmax()]
            # set kde to the name given
            x_kde, y_kde, model = univariate_kde(_X, bins, best_model_.name, kde_range=1e-3, smoothen_kde=smoothen_kde,
                                                 verbose=verbose, return_dist=True)
        elif hasattr(stats, kde):
            # fetches the kde if possible
            auto_fitted = None
            x_kde, y_kde, model = univariate_kde(_X, bins, kde, kde_range=1e-3, smoothen_kde=smoothen_kde,
                                                 verbose=verbose, return_dist=True)
        else:
            raise ValueError("kde value '{}' not found in scipy.stats".format(kde))

        # plot
        ax.plot(x_kde, y_kde, "-", color='r')
    else:
        auto_fitted = None
        model = None

    if x_label == "":
        x_label = _assign_x_label(title, X.name, kde is not None, auto_fitted, model)

    ax.set_xlabel(x_label)

    return ax
