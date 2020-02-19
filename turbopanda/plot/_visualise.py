#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# imports
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy import stats
from sklearn.metrics import r2_score

# locals
from turbopanda.utils import nearest_factors, belongs, fself, standardize
from turbopanda._metaml import MetaML
from turbopanda._pub_fig import save_figure


__all__ = ("scatter_grid", "missing", "hist_grid", "shape_multiplot")


def _iqr(a):
    """Calculate the IQR for an array of numbers."""
    return stats.scoreatpercentile(np.asarray(a), 75) - stats.scoreatpercentile(np.asarray(a), 25)


def _data_polynomial_length(length):
    # calculate length based on size of DF
    # dimensions follow this polynomial
    x = np.linspace(0, 250, 100)
    y = np.sqrt(np.linspace(0, 1, 100)) * 22 + 3
    p = np.poly1d(np.polyfit(x, y, deg=2))
    return int(p(length).round())


def _generate_square_like_grid(n, ax_size=2):
    """
    Given n, returns a fig, ax pairing of square-like grid objects
    """
    f1, f2 = nearest_factors(n, shape="square")
    fig, axes = plt.subplots(ncols=f1, nrows=f2, figsize=(ax_size * f1, ax_size * f2))
    if axes.ndim > 1:
        axes = list(it.chain.from_iterable(axes))
    return fig, axes


def _generate_diag_like_grid(n, direction, ax_size=2):
    """ Direction is in [row, column]"""
    belongs(direction, ["row", "column"])
    f1, f2 = nearest_factors(n, shape="diag")
    fmax, fmin = max(f1, f2), min(f1, f2)
    # get longest one
    tup, nc, nr = ((ax_size * fmin, ax_size * fmax), fmin, fmax) \
        if direction == 'row' else ((ax_size * fmax, ax_size * fmin), fmax, fmin)
    fig, axes = plt.subplots(ncols=nc, nrows=nr, figsize=tup)
    if axes.ndim > 1:
        axes = list(it.chain.from_iterable(axes))
    return fig, axes


def _freedman_diaconis_bins(a):
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


""" ################################### USEFUL FUNCTIONS ######################################"""


def shape_multiplot(n_plots: int, arrange: str = "square", ax_size: int = 2):
    """Determines the most optimal shape for a set of plots.

    Parameters
    ----------
    n_plots : int
        The total number of plots.
    arrange : str
        Choose from {'square', 'row' 'column'}. Indicates preference for direction
        of plots.
    ax_size : int
        The square size of each plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure
    axes : list of matplotlib.ax.Axes
        A list of axes to use.
    """
    if n_plots == 1:
        return plt.subplots(figsize=(ax_size, ax_size))
    else:
        return _generate_square_like_grid(n_plots, ax_size=ax_size) \
            if arrange == 'square' else _generate_diag_like_grid(n_plots, arrange, ax_size=ax_size)


def missing(mdf):
    """
    Plots the missing data as a greyscale heatmap.

    Parameters
    --------
    mdf : turb.MetaPanda
        The dataset

    Returns
    -------
    None
    """
    dims = (16, _data_polynomial_length(mdf.df_.shape[1]))

    # wrangle data
    out = mdf.df_.notnull().astype(np.int).T

    # figure out which plot we are using.!
    fig, ax = plt.subplots(figsize=dims)
    # use seaborn's heatmap
    ax.imshow(out, cmap="Greys", aspect="auto")
    # make sure to plot ALL labels. manual override
    ax.set_yticks(range(0, mdf.df_.shape[1], 2))
    ax.set_yticklabels(mdf.df_.columns)


def hist_grid(mdf, selector, arrange="square", savepath=None):
    """
    Plots a grid of histograms comparing the distributions in a MetaPanda
    selector.

    Parameters
    --------
    mdf : turb.MetaPanda
        The dataset
    selector : str or list/tuple of str
        Contains either types, meta column names, column names or regex-compliant strings
    arrange : str
        Choose from ['square', 'row', 'column']. Square arranges the plot as square-like as possible. Row
        prioritises plots row-like, and column-wise for column.
    savepath : None, bool, str
        saves the figure to file. If bool, uses the name in mdf, else uses given string. If None, no fig is saved.

    Returns
    -------
    None
    """
    belongs(arrange, ["square", "row", "column"])
    # get selector
    selection = mdf.view(selector)
    if selection.size > 0:
        fig, axes = shape_multiplot(len(selection), arrange)

        for i, x in enumerate(selection):
            # calculate the bins
            bins_ = min(_freedman_diaconis_bins(mdf.df_[x]), 50)
            axes[i].hist(mdf.df_[x].dropna(), bins=bins_)
            axes[i].set_title(x)
        fig.tight_layout()

        if isinstance(savepath, bool):
            save_figure(fig, "hist", mdf.name_)
        elif isinstance(savepath, str):
            save_figure(fig, "hist", mdf.name_, fp=savepath)


def scatter_grid(mdf, selector, target, arrange="square", savepath=None):
    """
    Plots a grid of scatter plots comparing each column for MetaPanda
    in selector to y target value.

    Parameters
    --------
    mdf : turb.MetaPanda
        The dataset
    selector : str or list/tuple of str
            Contains either types, meta column names, column names or regex-compliant strings
    target : str
        The y-response variable to plot
    arrange : str
        Choose from ['square', 'row', 'column']. Square arranges the plot as square-like as possible. Row
        prioritises plots row-like, and column-wise for column.
    savepath : None, bool, str
        saves the figure to file. If bool, uses the name in mdf, else uses given string.

    Returns
    -------
    None
    """
    belongs(arrange, ["square", "row", "column"])
    # get selector
    selection = mdf.view(selector)
    if selection.size > 0:
        fig, axes = shape_multiplot(len(selection), arrange)

        for i, x in enumerate(selection):
            axes[i].scatter(mdf.df_[x], mdf.df_[target], alpha=.5)
            # spearman correlation
            pair_corr = mdf.df_[[x, target]].corr(method="spearman").iloc[0, 1]
            axes[i].set_title("{}: r={:0.3f}".format(x, pair_corr))
        fig.tight_layout()

        if isinstance(savepath, bool):
            save_figure(fig, "scatter", mdf.name_)
        elif isinstance(savepath, str):
            save_figure(fig, "scatter", mdf.name_, fp=savepath)
