#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides plot_overview for data returned from `fit_basic`."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from turbopanda.corr import correlate
from turbopanda.corr._correlate import _row_to_matrix
from turbopanda.utils import union
from turbopanda.stats import vif, cook_distance
from turbopanda.plot import shape_multiplot
from ._clean import cleaned_subset


__all__ = ('coefficient_plot', 'overview_plot')


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


def coefficient_plot(plot, cv):
    """Plots the coefficients from a cv results."""
    coef = cv['w__']
    # sort the coefficients by size
    coef = coef.reindex(cv['w__'].mean(axis=0).sort_values().index, axis=1)
    # plot
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


def _basic_correlation_matrix(plot, df, x, y):
    # determine columns
    cols = union(df.view(x), y)
    # perform correlation
    corr = correlate(df[cols].dropna())
    # convert to matrix
    _cmatrix = _row_to_matrix(corr)
    # plot heatmap
    plot.pcolormesh(_cmatrix, vmin=-1., vmax=1., cmap="seismic")
    # if we have too many labels, randomly choose some
    if _cmatrix.shape[0] > 10:
        tick_locs = np.random.choice(_cmatrix.shape[0], 10, replace=False)
        plot.set_xticks(tick_locs)
        plot.set_xticklabels(_cmatrix.iloc[:, tick_locs].columns)
        plot.set_yticks(tick_locs)
        plot.set_yticklabels(_cmatrix.iloc[:, tick_locs].columns)

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


def _cooks(plot, df, x, y, yp):
    cooks = cook_distance(df, x, y, yp)
    plot.plot(cooks, 'ko', alpha=.6)
    plot.hlines(0, xmin=0, xmax=cooks.shape[0])
    plot.set_ylabel("Cook's distance")


def overview_plot(df, x, y, cv, yp):
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

    Returns
    -------
    a bunch of plots. No Return.
    """
    fig, ax = shape_multiplot(8, ax_size=4)
    # set yp as series
    yp = yp[y].squeeze()
    # pair them and remove NA
    _df = cleaned_subset(df, x, y)

    # plot 1. fitted vs. residual plots
    _fitted_vs_residual(ax[0], _df[y], yp)
    # plot 2. boxplot for scores
    _boxplot_scores(ax[1], cv)
    # plot 3. KDE plot estimation between y and yhat
    _actual_vs_predicted(ax[2], _df[y], yp)
    # plot 4. coefficient plot
    coefficient_plot(ax[3], cv)
    # plot 5. correlation matrix
    _basic_correlation_matrix(ax[4], df, x, y)
    # plot 6. variance inflation factor for each explanatory variable
    _plot_vif(ax[5], vif(df, x, y))
    # plot 7. q-q plot
    stats.probplot(_df[y], dist="norm", plot=ax[6])
    # plot 8. outlier detection using cook's distance plot
    """Cook's distance is defined as the sum of all the changes in the regression model when #
    observation i is removed from it."""
    _cooks(ax[7], df, x, y, yp)

    fig.tight_layout()
    plt.show()