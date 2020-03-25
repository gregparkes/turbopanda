#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles generic visualization/plotting functions."""

# future imports
from __future__ import absolute_import, division, print_function

import itertools as it
from typing import Optional, Union

import matplotlib.pyplot as plt
# imports
import numpy as np

from turbopanda.utils import belongs, remove_na, instance_check, difference
from ._gridplot import gridplot
from ._histogram import histogram
from ._save_fig import save

__all__ = ("scatter_grid", "missing", "hist_grid")


def missing(mdf: "MetaPanda"):
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

    def _data_polynomial_length(length):
        # calculate length based on size of DF
        # dimensions follow this polynomial
        x = np.linspace(0, 250, 100)
        y = np.sqrt(np.linspace(0, 1, 100)) * 22 + 3
        p = np.poly1d(np.polyfit(x, y, deg=2))
        return int(p(length).round())

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


def hist_grid(mdf: "MetaPanda",
              selector,
              arrange: str = "square",
              plot_size: int = 3,
              shared_dist: str = "auto",
              savepath: Optional[Union[str, bool]] = None,
              **hist_kws):
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
    plot_size : int, default=3
        The size of each axes
    shared_dist : str/tuple of str/dict, default="auto"
        Determines what KDE to fit to the data, set to None if you don't want
        If tuple/list: attempts using these specified distributions
        If dict: maps column name (k) to distribution choice (v)
    savepath : None, bool, str
        saves the figure to file. If bool, uses the name in mdf, else uses given string. If None, no fig is saved.

    Other Parameters
    ----------------
    hist_kws : dict
        Keywords to pass to `turb.plot.histogram`

    Returns
    -------
    None
    """
    instance_check(plot_size, int)
    instance_check(shared_dist, (type(None), str, list, tuple, dict))
    instance_check(savepath, (type(None), str, bool))
    belongs(arrange, ["square", "row", "column"])
    # get selector
    selection = mdf.view(selector)
    if selection.size > 0:
        fig, axes = gridplot(len(selection), arrange, ax_size=plot_size)

        if not isinstance(shared_dist, dict):
            for i, x in enumerate(selection):
                _ = histogram(mdf[x].dropna(), ax=axes[i], title=x, kde=shared_dist, **hist_kws)
            fig.tight_layout()
        else:
            for i, (x, d) in enumerate(shared_dist.items()):
                _ = histogram(mdf[x].dropna(), ax=axes[i], title=x, kde=d, **hist_kws)
            # iterate over any 'remaining' columns in selection and handle appropriately
            remaining = difference(selection, tuple(shared_dist.keys()))
            if remaining.shape[0] > 0:
                for i, x in enumerate(remaining):
                    _ = histogram(mdf[x].dropna(), ax=axes[i+len(shared_dist)], title=x, kde="auto", **hist_kws)
            fig.tight_layout()

        if isinstance(savepath, bool):
            save(fig, "hist", mdf.name_)
        elif isinstance(savepath, str):
            save(fig, "hist", mdf.name_, fp=savepath)


def scatter_grid(mdf: "MetaPanda",
                 x,
                 y,
                 arrange: str = "square",
                 plot_size: int = 3,
                 savepath: Optional[Union[bool, str]] = None):
    """
    Plots a grid of scatter plots comparing each column for MetaPanda
    in selector to y target value.

    Parameters
    --------
    mdf : turb.MetaPanda
        The dataset
    x : str or list/tuple of str
            Contains either types, meta column names, column names or regex-compliant strings
    y : str or list/tuple of str
            Contains either types, meta column names, column names or regex-compliant strings
    arrange : str
        Choose from ['square', 'row', 'column']. Square arranges the plot as square-like as possible. Row
        prioritises plots row-like, and column-wise for column.
    plot_size : int
        The size of each axes
    savepath : None, bool, str
        saves the figure to file. If bool, uses the name in mdf, else uses given string.

    Returns
    -------
    None
    """
    from turbopanda.corr import bicorr

    belongs(arrange, ["square", "row", "column"])
    # get selector
    x_sel = mdf.view(x)
    y_sel = mdf.view(y)
    # create a product between x and y and plot
    prod = list(it.product(x_sel, y_sel))

    if len(prod) > 0:
        fig, axes = gridplot(len(prod), arrange, ax_size=plot_size)
        for i, (_x, _y) in enumerate(prod):
            # pair x, y
            __x, __y = remove_na(mdf[_x].values, mdf[_y].values, paired=True)
            axes[i].scatter(__x.flatten(), __y, alpha=.5)
            # line of best fit
            xn = np.linspace(__x.min(), __x.max(), 100)
            z = np.polyfit(__x.flatten(), __y, deg=1)
            axes[i].plot(xn, np.polyval(z, xn), 'k--')
            # spearman correlation
            pair_corr = bicorr(mdf[_x], mdf[_y]).loc['spearman', 'r']
            axes[i].set_title("r={:0.3f}".format(pair_corr))
            axes[i].set_xlabel(_x)
            axes[i].set_ylabel(_y)

        fig.tight_layout()

        if isinstance(savepath, bool):
            save(fig, "scatter", mdf.name_)
        elif isinstance(savepath, str):
            save(fig, "scatter", mdf.name_, fp=savepath)
