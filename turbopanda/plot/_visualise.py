#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handles generic visualization/plotting functions."""

# future imports
from __future__ import absolute_import, division, print_function

import itertools as it
from typing import Optional, Union

# imports
import numpy as np
from pandas import DataFrame

from turbopanda._metapanda import MetaPanda, SelectorType
from turbopanda.utils import belongs, remove_na, instance_check, difference, nonnegative

from ._gridplot import gridplot
from ._histogram import histogram
from ._save_fig import save

__all__ = ("scatter_grid", "hist_grid")


def hist_grid(
    mdf: Union[DataFrame, "MetaPanda"],
    subset: SelectorType,
    arrange: str = "square",
    plot_size: int = 3,
    shared_dist: str = "auto",
    savepath: Optional[Union[str, bool]] = None,
    **hist_kws
):
    """
    Plots a grid of histograms comparing the distributions in a MetaPanda
    selector.

    Parameters
    --------
    mdf : turb.MetaPanda
        The dataset
    subset : str or list/tuple of str
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
    # checks
    instance_check(shared_dist, (type(None), str, list, tuple, dict))
    instance_check(savepath, (type(None), str, bool))
    nonnegative(plot_size, int)
    belongs(arrange, ["square", "row", "column"])
    # make a metapanda if we have a dataframe.
    _mdf = MetaPanda(mdf) if isinstance(mdf, DataFrame) else mdf

    # get selector
    selection = _mdf.view(subset)
    # assuming we've selected something...
    if selection.size > 0:
        fig, axes = gridplot(len(selection), arrange, ax_size=plot_size)

        if not isinstance(shared_dist, dict):
            for i, x in enumerate(selection):
                _ = histogram(
                    _mdf[x].dropna(), ax=axes[i], title=x, kde=shared_dist, **hist_kws
                )
            fig.tight_layout()
        else:
            for i, (x, d) in enumerate(shared_dist.items()):
                _ = histogram(_mdf[x].dropna(), ax=axes[i], title=x, kde=d, **hist_kws)
            # iterate over any 'remaining' columns in selection and handle appropriately
            remaining = difference(selection, tuple(shared_dist.keys()))
            if remaining.shape[0] > 0:
                for i, x in enumerate(remaining):
                    _ = histogram(
                        _mdf[x].dropna(),
                        ax=axes[i + len(shared_dist)],
                        title=x,
                        kde="auto",
                        **hist_kws
                    )
            fig.tight_layout()

        if isinstance(savepath, bool):
            save(fig, "hist", _mdf.name_)
        elif isinstance(savepath, str):
            save(fig, "hist", _mdf.name_, fp=savepath)


def scatter_grid(
    mdf: Union[DataFrame, "MetaPanda"],
    x: SelectorType,
    y: SelectorType,
    arrange: str = "square",
    plot_size: int = 3,
    best_fit: bool = True,
    best_fit_deg: int = 1,
    savepath: Optional[Union[bool, str]] = None,
):
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
    best_fit : bool
        If True, draws a line of best fit
    best_fit_deg : int, default=1
        The degree of the line of best fit, can draw polynomial
    savepath : None, bool, str
        saves the figure to file. If bool, uses the name in mdf, else uses given string.

    Returns
    -------
    None
    """
    from turbopanda.corr import bicorr

    # checks
    instance_check((plot_size, best_fit_deg), int)
    instance_check(savepath, (type(None), str, bool))
    instance_check(best_fit, bool)
    nonnegative(
        (
            best_fit_deg,
            plot_size,
        )
    )
    belongs(arrange, ["square", "row", "column"])

    # make a metapanda if we have a dataframe.
    _mdf = MetaPanda(mdf) if isinstance(mdf, DataFrame) else mdf

    # get selector
    x_sel = _mdf.view(x)
    y_sel = _mdf.view(y)
    # create a product between x and y and plot
    prod = list(it.product(x_sel, y_sel))

    if len(prod) > 0:
        fig, axes = gridplot(len(prod), arrange, ax_size=plot_size)
        for i, (_x, _y) in enumerate(prod):
            # pair x, y
            __x, __y = remove_na(_mdf[_x].values, _mdf[_y].values, paired=True)
            axes[i].scatter(__x.flatten(), __y, alpha=0.5)
            # line of best fit
            if best_fit:
                xn = np.linspace(__x.min(), __x.max(), 100)
                z = np.polyfit(__x.flatten(), __y, deg=best_fit_deg)
                axes[i].plot(xn, np.polyval(z, xn), "k--")

            # spearman correlation
            pair_corr = bicorr(_mdf[_x], _mdf[_y]).loc["spearman", "r"]
            axes[i].set_title("r={:0.3f}".format(pair_corr))
            axes[i].set_xlabel(_x)
            axes[i].set_ylabel(_y)

        fig.tight_layout()

        if isinstance(savepath, bool):
            save(fig, "scatter", _mdf.name_)
        elif isinstance(savepath, str):
            save(fig, "scatter", _mdf.name_, fp=savepath)
