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
from .utils import nearest_factors, belongs, fself, standardize
from .metaml import MetaML
from ._pub_fig import save_figure


__all__ = ("plot_scatter_grid", "plot_missing", "plot_hist_grid",
           "plot_coefficients", "plot_actual_vs_predicted")


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
    tup, nc, nr = ((ax_size*fmin, ax_size*fmax), fmin, fmax) \
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


def plot_missing(mdf):
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


def plot_hist_grid(mdf, selector, arrange="square", savepath=None):
    """
    Plots a grid of histograms comparing the distributions in a MetaPanda
    selector.

    Parameters
    --------
    mdf : turb.MetaPanda
        The dataset
    selector : str or list/tuple of str
        Contains either custypes.py, meta column names, column names or regex-compliant strings
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
        # gets grid-like coordinates for our selector length.
        fig, axes = _generate_square_like_grid(len(selection)) \
            if arrange == 'square' else _generate_diag_like_grid(len(selection), arrange)

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


def plot_scatter_grid(mdf, selector, target, arrange="square", savepath=None):
    """
    Plots a grid of scatter plots comparing each column for MetaPanda
    in selector to y target value.

    Parameters
    --------
    mdf : turb.MetaPanda
        The dataset
    selector : str or list/tuple of str
            Contains either custypes.py, meta column names, column names or regex-compliant strings
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
        fig, axes = _generate_square_like_grid(len(selection)) \
            if arrange == 'square' else _generate_diag_like_grid(len(selection), arrange)

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


def plot_actual_vs_predicted(mml):
    """
    Plots the actual (regression) values against the predicted values. If there are multiple,
    creates a multiplot.

    Parameters
    --------
    mml : MetaML
        The fitted machine learning model(s).

    Returns
    -------
    None
    """
    if isinstance(mml, MetaML):
        if mml.is_fit:
            if mml.multioutput:
                fig, axes = plt.subplots(ncols=mml.y.shape[1], figsize=(3 * mml.y.shape[1], 4))
                for a, col in zip(axes, mml.y_names):
                    min_y, max_y = mml.y[col].min(), mml.y[col].max()
                    a.scatter(mml.y[col], mml.yp[col], alpha=.3, marker="x",
                              label=r"$r={:0.3f}$".format(r2_score(mml.y[col], mml.yp[col])))
                    a.plot([min_y, max_y], [min_y, max_y], 'k-')
                    a.set_title(col)
            else:
                fig, axes = plt.subplots(figsize=(8, 4))
                min_y, max_y = mml.y.min(), mml.y.max()
                axes.scatter(mml.y, mml.yp, alpha=.3, marker="x", label=r"$r={:0.3f}$".format(mml.score_r2))
                axes.plot([min_y, max_y], [min_y, max_y], 'k-')
                axes.set_title(mml.y_names)


def plot_coefficients(mml, normalize=False, use_absolute=False, drop_intercept=True):
    """
    Plots the coefficients from a fitted machine learning model using MetaML.

    Parameters
    --------
    mml : MetaML or list of MetaML
        The fitted machine learning model(s). If a list, each MetaML must have
        the same data (columns)
    normalize : bool
        Use standardization on the values if True
    use_absolute : bool
        If True, uses absolute value of coefficients instead.
    drop_intercept : bool
        If True, drops the intercept row from mml.coef_mat

    Returns
    -------
    None
    """
    # assumes the coef_mat variable is present

    f_apply = np.abs if use_absolute else fself
    norm_apply = standardize if normalize else fself

    def _plot_single_box(model, ax, index):
        if isinstance(model, MetaML):
            if model.is_fit and hasattr(model, "coef_mat"):
                # new order
                transformed = model.coef_mat.apply(f_apply).apply(norm_apply)
                if drop_intercept and ("intercept" in transformed.index):
                    # intercept should be on index
                    transformed.drop("intercept", axis=0, inplace=True)
                no = transformed.mean(axis=1).sort_values().index
                buf = len(model.coef_mat) / 20
                # perform transformations and get data
                data = transformed.reindex(no)
                if index == 0:
                    ax.boxplot(data, vert=False, labels=no)
                else:
                    ax.boxplot(data, vert=False)
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                ax.set_ylim(-buf, len(model.coef_mat) + buf)
                ax.vlines([0], ymin=0, ymax=len(model.coef_mat), linestyle="--", color="red")
                ax.set_xlabel("Coefficients")
                ax.set_title("Model: {}".format(model.model_str))
                ax.margins(y=0)

    if isinstance(mml, MetaML):
        fig, axes = plt.subplots(figsize=(5, _data_polynomial_length(mml.coef_mat.shape[0])))
        _plot_single_box(mml, axes, 0)
    elif isinstance(mml, (list, tuple)):
        fig, axes = plt.subplots(ncols=len(mml),
                                 figsize=(4 * len(mml), _data_polynomial_length(mml[0].coef_mat.shape[0])))
        for i, (m, a) in enumerate(zip(mml, axes)):
            _plot_single_box(m, a, i)
    else:
        raise TypeError("mml type '{}' not recognised".format(type(mml)))
    fig.tight_layout()
    plt.show()
