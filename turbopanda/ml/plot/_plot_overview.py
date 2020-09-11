#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides plot_overview for data returned from `fit_basic`."""

import itertools as it
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from turbopanda._metapanda import SelectorType

from turbopanda.str import shorten
from turbopanda.pipe import absolute
from turbopanda.plot import gridplot, bibox1d, widebox
from turbopanda.utils import instance_check, intersect
from turbopanda.ml._clean import select_xcols, preprocess_continuous_X_y

__all__ = ('coefficient', 'overview')


def _is_coefficients(cv):
    return cv["w__"].shape[1] > 0


def _fitted_vs_residual(plot, y, yp):
    plot.scatter(yp, y - yp, color='g', alpha=.5)
    plot.hlines(0, xmin=yp.min(), xmax=yp.max(), color='r')
    plot.set_xlabel("Fitted Values")
    plot.set_ylabel("Residuals")


def _boxplot_scores(plot, cv, score="RMSE"):
    # use bibox1d
    _data = cv.df_.pipe(absolute, 'train_score|test_score')
    bibox1d(_data['test_score'], _data['train_score'],
            measured=score, vertical=False,
            ax=plot, mannwhitney=False)


def _actual_vs_predicted(plot, y, yp):
    # KDE plot estimation between y and yhat
    plot.scatter(y, yp, color='b', alpha=.5)
    plot.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], 'k-')
    plot.set_xlabel("Actual Values")
    plot.set_ylabel("Fitted Values")


def coefficient(cv, plot=None):
    """Plots the coefficients from a cv results."""
    if _is_coefficients(cv):
        coef = cv['w__']
        # sort the coefficients by size
        coef = coef.reindex(cv['w__'].mean(axis=0).sort_values().index, axis=1)
        # rename the columns

        # plot
        if plot is None:
            fig, plot = plt.subplots(figsize=(8, 5))

        widebox(coef, vert=False, outliers=False,
                measured=r"$\beta$", ax=plot)
        plot.set_title("Coefficients")


def _basic_correlation_matrix(plot, _cmatrix):
    # plot heatmap
    plot.imshow(_cmatrix, vmin=-1., vmax=1., cmap="seismic")
    # if we have too many labels, randomly choose some
    if _cmatrix.shape[0] > 10:
        tick_locs = np.random.choice(_cmatrix.shape[0], 10, replace=False)
        plot.set_xticks(tick_locs)
        plot.set_xticklabels(_cmatrix.iloc[:, tick_locs].columns)
        plot.set_yticks(tick_locs)
        plot.set_yticklabels(_cmatrix.iloc[:, tick_locs].columns)
    else:
        plot.set_xticks(range(_cmatrix.shape[0]))
        plot.set_xticklabels(_cmatrix.columns)
        plot.set_yticks(range(_cmatrix.shape[0]))
        plot.set_yticklabels(_cmatrix.columns)

    plt.setp(plot.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plot.tick_params('x', rotation=45)
    for tick in plot.get_xmajorticklabels():
        tick.set_horizontalalignment('right')

def _cooks(plot, cooks):
    plot.plot(cooks, 'ko', alpha=.6)
    plot.hlines(0, xmin=0, xmax=cooks.shape[0])
    plot.set_ylabel("Cook's distance")


""" ######################### PUBLIC FUNCTIONS ######################### """


def overview(df: "MetaPanda",
             x: SelectorType,
             y: str,
             cv: "MetaPanda",
             yp: "MetaPanda",
             plot_names: Optional[List[str]] = None,
             plot_size: int = 3):
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
            'correlation', 'qqplot', 'cooks'}
        If None: draws ALL.
    plot_size : int, optional
        Defines the size of each plot size.

    Returns
    -------
    a bunch of plots. No Return.
    """
    from turbopanda.corr._correlate import correlate, row_to_matrix
    from turbopanda.stats import cook_distance

    instance_check(plot_names, (type(None), tuple))
    instance_check(y, str)
    options_ = ('resid_fitted', 'score', 'actual_predicted', 'coef',
                'correlation', 'qqplot', 'cooks')

    """ Prepare data here. """
    # set yp as series
    yp = yp[y].squeeze()
    # pair them and remove NA
    _df = df.df_ if not isinstance(df, pd.DataFrame) else df
    _xcols = select_xcols(_df, x, y)
    _x, _y = preprocess_continuous_X_y(_df, _xcols, y)

    options_yes_ = (True, True, True, True, 1 < len(_xcols) < 50,
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
        coefficient(cv, ax[next(I)])
    if "correlation" in overlap_ and (1 < len(_xcols) < 50):
        # plot 5. correlation matrix
        corr = correlate(df, x)
        _cmatrix = row_to_matrix(corr)
        _basic_correlation_matrix(ax[next(I)], _cmatrix)
    if "qqplot" in overlap_:
        # plot 7. q-q plot
        stats.probplot(_df[y], dist="norm", plot=ax[next(I)])
    if "cooks" in overlap_:
        # plot 8. outlier detection using cook's distance plot
        _c = cook_distance(df, x, y, yp)
        _cooks(ax[next(I)], _c)

    fig.tight_layout()
    plt.show()
