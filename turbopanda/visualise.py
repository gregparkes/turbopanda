#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:37:45 2019

@author: gparkes
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy import stats

from .utils import nearest_square_factors
from .metaml import MetaML


__all__ = ["plot_scatter_grid", "plot_missing", "plot_hist_grid",
           "plot_coefficients"]


def _iqr(a):
    """Calculate the IQR for an array of numbers."""
    a = np.asarray(a)
    q1 = stats.scoreatpercentile(a, 25)
    q3 = stats.scoreatpercentile(a, 75)
    return q3 - q1


def _data_polynomial_length(L):
    # calculate length based on size of DF
    # dimensions follow this polynomial
    X = np.linspace(0, 250, 100)
    Y = np.linspace(0, 1, 100)**(1/2) * 22 + 3
    P = np.poly1d(np.polyfit(X,Y,deg=2))
    return int(P(L).round())


def _generate_square_like_grid(n, square_size=2):
    """
    Given n, returns a fig, ax pairing of square-like grid objects
    """
    f1,f2 = nearest_square_factors(n)
    fig, axes = plt.subplots(ncols=f1, nrows=f2, figsize=(square_size*f1, square_size*f2))
    if (axes.ndim > 1):
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


def plot_missing(mdf):
    """
    Plots the missing data as a greyscale heatmap.
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


def plot_hist_grid(mdf, selector):
    """
    Plots a grid of histograms comparing the distributions in a MetaPanda
    selector.

    Parameters
    --------
    mdf : turb.MetaPanda
        The dataset
    selector : str or list/tuple of str
            Contains either types, meta column names, column names or regex-compliant strings

    Returns
    -------
    None
    """
    # get selector
    selection = mdf.view(selector)
    if selection.size > 0:
        # gets grid-like coordinates for our selector length.
        fig, axes = _generate_square_like_grid(len(selection))
        for i, x in enumerate(selection):
            # calculate the bins
            bins_ = min(_freedman_diaconis_bins(mdf.df_[x]), 50)
            axes[i].hist(mdf.df_[x].dropna(), bins=bins_)
            axes[i].set_title(x)
        fig.tight_layout()


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
        fig, axes = _generate_square_like_grid(len(selection))
        for i, x in enumerate(selection):
            axes[i].scatter(mdf.df_[x], mdf.df_[target], alpha=.5)
            # spearman correlation
            pair_corr = mdf.df_[[x, target]].corr(method="spearman").iloc[0, 1]
            axes[i].set_title("{}: r={:0.3f}".format(x, pair_corr))
        fig.tight_layout()


def plot_coefficients(mml):
    """
    Plots the coefficients from a fitted machine learning model using MetaML.

    Parameters
    --------
    mml : MetaML or list of MetaML
        The fitted machine learning model(s). If a list, each MetaML must have
        the same data (columns)

    Returns
    -------
    None
    """
    # assumes the coef_mat variable is present

    def _plot_single_box(m, ax, i):
        if isinstance(m, MetaML):
            if m.fit:
                # new order
                no = m.coef_mat.mean(axis=1).sort_values().index
                buffer = len(m.coef_mat) / 20

                if i==0:
                    ax.boxplot(m.coef_mat.reindex(no), vert=False, labels=no)
                else:
                    ax.boxplot(m.coef_mat.reindex(no), vert=False)
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                ax.set_ylim(-buffer, len(m.coef_mat)+buffer)
                ax.vlines([0], ymin=0, ymax=len(m.coef_mat), linestyle="--", color="red")
                ax.set_xlabel("Coefficients")
                ax.set_title("Model: {}".format(m.model_str))

    if isinstance(mml, MetaML):
        fig, ax = plt.subplots(figsize=(5, _data_polynomial_length(mml.coef_mat.shape[0])))
        _plot_single_box(mml, ax, 0)
    elif isinstance(mml, (list, tuple)):
        fig, ax = plt.subplots(ncols=len(mml), figsize=(4*len(mml), _data_polynomial_length(mml[0].coef_mat.shape[0])))
        for i, (m, a) in enumerate(zip(mml, ax)):
            _plot_single_box(m, a, i)

    fig.tight_layout()
    plt.show()
