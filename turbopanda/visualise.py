#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:37:45 2019

@author: gparkes
"""
import matplotlib.pyplot as plt
import itertools as it

from .utils import nearest_square_factors


__all__ = ["plot_scatter_grid"]


def _generate_square_like_grid(n):
    """
    Given n, returns a fig, ax pairing of square-like grid objects
    """
    f1,f2 = nearest_square_factors(n)
    fig, axes = plt.subplots(ncols=f1, nrows=f2, figsize=(3*f1, 3*f2))
    if (axes.ndim > 1):
        axes = list(it.chain.from_iterable(axes))
    return fig, axes


def plot_scatter_grid(mdf, selector, target):
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

    Returns
    -------
    None
    """
    # get selector
    selection = mdf.view(selector)
    if selection.size > 0:
        # gets grid-like coordinates for our selector length.
        fig, axes = _generate_square_like_grid(len(selector))
        for i, x in enumerate(selection):
            axes[i].scatter(mdf.df_[x], mdf.df_[target], alpha=.5)
            # spearman correlation
            pair_corr = mdf.df_[[x, target]].corr(method="spearman").iloc[0, 1]
            axes[i].set_title("{}: r={:0.3f}".format(x, pair_corr))
        fig.tight_layout()
