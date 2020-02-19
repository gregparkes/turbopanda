#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides basic plots for machine-learning results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from turbopanda.corr import correlate
from turbopanda.corr._correlate import _row_to_matrix
from turbopanda.plot import shape_multiplot
from turbopanda.utils import union

from ._clean import cleaned_subset
from turbopanda.stats import vif, cook_distance


__all__ = ('coefficient', 'overview_plot')


def _fitted_vs_residual(plot, y, yp):
    plot.scatter(yp, y - yp, color='g', alpha=.5)
    plot.hlines(0, xmin=yp.min(), xmax=yp.max(), color='r')
    plot.set_xlabel("Fitted Values")
    plot.set_ylabel("Residuals")


def _boxplot_scores(plot, cv):
    plot.boxplot(cv['train_score|test_score'].values)
    plot.set_xlabel("Train/Test Score")
    plot.set_ylabel(r"$r^2$")
    plot.tick_params('x', rotation=45)
    plot.set_xticks(range(1, 3))
    plot.set_xticklabels(['test', 'train'])


def _actual_vs_predicted(plot, y, yp, kde=True):
    # KDE plot estimation between y and yhat
    if kde:
        sns.kdeplot(y, yp, color='b', ax=plot, n_levels=10)
    else:
        plot.scatter(y, yp, color='b', alpha=.5)

    plot.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], 'k-')
    # sns.distplot(yp, color='k', ax=ax[0, 2], norm_hist=True)
    plot.set_xlabel("Actual Values")
    plot.set_ylabel("Fitted Values")


def coefficient(plot, cv):
    """Plots the coefficients from a cv results."""
    coef = cv['w__']
    plot.boxplot(coef.values)
    plot.set_xlabel("Coefficient")

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


def _basic_correlation_matrix(plot, df, x, y):
    cols = union(df.view(x), y)
    corr = correlate(df[cols].dropna())
    _cmatrix = _row_to_matrix(corr)
    sns.heatmap(_cmatrix, ax=plot, cmap="seismic", vmin=-1., vmax=1., square=True, center=0, cbar=None)
    # if we have too many labels, randomly choose some
    if _cmatrix.shape[0] > 10:
        tick_locs = np.random.choice(_cmatrix.shape[0], 10, replace=False)
        plot.set_xticks(tick_locs)
        plot.set_xticklabels(_cmatrix.iloc[:, tick_locs].columns)


def _plot_vif(plot, _vif):
    if isinstance(_vif, pd.Series):
        plot.bar(range(1, len(_vif) + 1), _vif.values, width=.7, color='r')
        plot.set_xlabel("Feature")
        plot.set_ylabel("VIF")
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
    x : selector
    y : str
    cv : MetaPanda
    yp : pd.Series

    Returns
    -------
    a bunch of plots. No Return.
    """
    fig, ax = shape_multiplot(8, ax_size=3)

    # pair them and remove NA
    _df = cleaned_subset(df, x, y)

    # plot 1. fitted vs. residual plots
    _fitted_vs_residual(ax[0], _df[y], yp)
    # plot 2. boxplot for scores
    _boxplot_scores(ax[1], cv)
    # plot 3. KDE plot estimation between y and yhat
    _actual_vs_predicted(ax[2], _df[y], yp, kde=False)
    # plot 4. coefficient plot
    coefficient(ax[3], cv)
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
