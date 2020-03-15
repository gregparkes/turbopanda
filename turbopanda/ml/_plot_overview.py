#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides plot_overview for data returned from `fit_basic`."""

import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from turbopanda.corr import correlate
from turbopanda.corr._correlate import _row_to_matrix
from turbopanda.utils import union, instance_check, intersect
from turbopanda.stats import vif, cook_distance
from turbopanda.plot import gridplot
from turbopanda._dependency import is_statsmodels_installed
from ._clean import ml_ready

__all__ = ('coefficient_plot', 'overview_plot')


def _is_coefficients(cv):
    return cv["w__"].shape[1] > 0


def _fitted_vs_residual(plot, y, yp):
    plot.scatter(yp, y - yp, color='g', alpha=.5)
    plot.hlines(0, xmin=yp.min(), xmax=yp.max(), color='r')
    plot.set_xlabel("Fitted Values")
    plot.set_ylabel("Residuals")


def _boxplot_scores(plot, cv, score="RMSE"):
    # has columns: fit_time, test_score, train_score
    # create a copy
    res = cv.copy()
    # transform.
    if res['test_score'].mean() < 0.:
        res.transform(np.abs, "_score")

    plot.boxplot(res['train_score|test_score'].values)
    plot.set_xlabel("Train/Test Score")
    plot.set_ylabel(score)
    plot.set_title("{}: {:0.3f}".format(score, res['test_score'].median()))
    plot.tick_params('x', rotation=45)
    plot.set_xticks(range(1, 3))
    plot.set_xticklabels(['test', 'train'])


def _actual_vs_predicted(plot, y, yp):
    # KDE plot estimation between y and yhat
    plot.scatter(y, yp, color='b', alpha=.5)
    plot.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], 'k-')
    plot.set_xlabel("Actual Values")
    plot.set_ylabel("Fitted Values")


def coefficient_plot(cv, plot=None):
    """Plots the coefficients from a cv results."""
    if _is_coefficients(cv):
        coef = cv['w__']
        # sort the coefficients by size
        coef = coef.reindex(cv['w__'].mean(axis=0).sort_values().index, axis=1)
        # plot
        if plot is None:
            fig, plot = plt.subplots(figsize=(8, 5))

        plot.boxplot(coef.values)
        plot.set_yscale("log")
        plot.set_ylabel(r"$\beta$")
        # if we have too many labels, randomly select some
        if len(coef.columns) > 10:
            # subset
            xtick_locs = np.random.choice(len(coef.columns), 10, replace=False)
            plot.set_xticks(xtick_locs)
            plot.set_xticklabels(coef.iloc[:, xtick_locs].columns.str[3:])
        else:
            plot.set_xticks(range(1, len(coef.columns) + 1))
            plot.set_xticklabels(coef.columns.str[3:])

        plot.tick_params("x", rotation=45)
        for tick in plot.get_xmajorticklabels():
            tick.set_horizontalalignment('right')
        plot.set_title("Coefficients")


def _basic_correlation_matrix(plot, _cmatrix):
    # plot heatmap
    plot.pcolormesh(_cmatrix, vmin=-1., vmax=1., cmap="seismic")
    # if we have too many labels, randomly choose some
    if _cmatrix.shape[0] > 10:
        tick_locs = np.random.choice(_cmatrix.shape[0], 10, replace=False)
        plot.set_xticks(tick_locs)
        plot.set_xticklabels(_cmatrix.iloc[:, tick_locs].columns)
        plot.set_yticks(tick_locs)
        plot.set_yticklabels(_cmatrix.iloc[:, tick_locs].columns)
    else:
        plot.set_xticks(range(1,_cmatrix.shape[0]+1))
        plot.set_xticklabels(_cmatrix.columns)
        plot.set_yticks(range(1,_cmatrix.shape[0]+1))
        plot.set_yticklabels(_cmatrix.columns)

    plot.tick_params('x', rotation=45)
    for tick in plot.get_xmajorticklabels():
        tick.set_horizontalalignment('right')


def _plot_vif(plot, _vif):
    if isinstance(_vif, pd.Series):
        plot.bar(range(1, len(_vif) + 1), _vif.values, width=.7, color='r')
        plot.set_xlabel("Feature")
        plot.set_ylabel("VIF")
        plot.set_yscale("log")
        if len(_vif) > 10:
            tick_locs = np.random.choice(len(_vif), 10, replace=False)
            plot.set_xticks(tick_locs)
            plot.set_xticklabels(_vif.iloc[tick_locs].index.values)
        else:
            plot.set_xticks(range(1, len(_vif) + 1))
            plot.set_xticklabels(_vif.index.values)

        plot.tick_params('x', rotation=45)


def _cooks(plot, cooks):
    plot.plot(cooks, 'ko', alpha=.6)
    plot.hlines(0, xmin=0, xmax=cooks.shape[0])
    plot.set_ylabel("Cook's distance")


""" ######################### PUBLIC FUNCTIONS ######################### """


def overview_plot(df, x, y, cv, yp, plot_names=None, plot_size=3):
    """Presents an overview of the results of a machine learning basic run.

    Parameters
    ----------
    df : MetaPanda
        The raw dataset.
    x : selector
        The input selection to the model
    y : str
        The target vector
    cv : MetaPanda
        The results cv from a call to `fit_basic`
    yp : MetaPanda
        The result fitted values from a call to `fit_basic`
    plot_names : list of str, optional
        Names of specific plot types to draw.
        Choose any combo of: {'resid_fitted', 'score', 'actual_predicted', 'coef',
            'correlation', 'vif', 'qqplot', 'cooks'}
        If None: draws ALL.
    plot_size : int, optional
        Defines the size of each plot size.

    Returns
    -------
    a bunch of plots. No Return.
    """
    instance_check(plot_names, (type(None), tuple))
    instance_check(y, str)
    options_ = ('resid_fitted', 'score', 'actual_predicted', 'coef',
                'correlation', 'vif', 'qqplot', 'cooks')

    """ Prepare data here. """
    # set yp as series
    yp = yp[y].squeeze()
    # pair them and remove NA
    _df, _x, _y, _xcols = ml_ready(df, x, y)
    options_yes_ = (True, True, True, True, 1 < len(_xcols) < 50, (len(_xcols) > 1) and is_statsmodels_installed(),
                    True, True)
    # compress down options
    option_compressed = list(it.compress(options_, options_yes_))

    """ Make plots here """
    if plot_names is None:
        plot_names = options_
    # overlap plots
    overlap_ = sorted(intersect(option_compressed, plot_names),
                      key=options_.index)

    # make plots
    fig, ax = gridplot(len(overlap_), ax_size=plot_size)
    I = it.count()

    if "score" in overlap_:
        # plot 2. boxplot for scores
        _boxplot_scores(ax[next(I)], cv)
    if "resid_fitted" in overlap_:
        # plot 1. fitted vs. residual plots
        _fitted_vs_residual(ax[next(I)], _y, yp)
    if "actual_predicted" in overlap_:
        # plot 3. KDE plot estimation between y and yhat
        _actual_vs_predicted(ax[next(I)], _y, yp)
    if "coef" in overlap_:
        # plot 4. coefficient plot
        coefficient_plot(cv, ax[next(I)])
    if "correlation" in overlap_ and (1 < len(_xcols) < 50):
        # plot 5. correlation matrix
        corr = correlate(df, x)
        _cmatrix = _row_to_matrix(corr)
        _basic_correlation_matrix(ax[next(I)], _cmatrix)
    if "vif" in overlap_ and len(_xcols) > 1:
        # plot 6. variance inflation factor for each explanatory variable
        _v = vif(df, x, y)
        _plot_vif(ax[next(I)], _v)
    if "qqplot" in overlap_:
        # plot 7. q-q plot
        stats.probplot(_df[y], dist="norm", plot=ax[next(I)])
    if "cooks" in overlap_:
        # plot 8. outlier detection using cook's distance plot
        _c = cook_distance(df, x, y, yp)
        _cooks(ax[next(I)], _c)

    fig.tight_layout()
    plt.show()
